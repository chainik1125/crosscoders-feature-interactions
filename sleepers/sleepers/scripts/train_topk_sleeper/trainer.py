
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from typing import Any
from wandb.sdk.wandb_run import Run
from model_diffing.models.activations.topk import TopkActivation
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.utils import (
	calculate_explained_variance_X,
	calculate_reconstruction_loss,
)
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from model_diffing.data.model_hookpoint_dataloader import BaseModelHookpointActivationsDataloader
from sleepers.scripts.utils import calculate_fvu_X
from sleepers.scripts.utils import sharpness_func, add_penalty, get_neuron_preacts_cutoff, neuron_sharpness_quick
from sleepers.scripts.llms import build_llm_lora

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables - will be set dynamically by the trainer
W_ins = None
b_ins = None 
W_outs = None
b_outs = None

class TopKTrainer(BaseModelHookpointTrainer[BaseTrainConfig, TopkActivation]):
	def __init__(
		self,
		cfg: BaseTrainConfig,
		activations_dataloader: BaseModelHookpointActivationsDataloader,
		crosscoder: AcausalCrosscoder[TopkActivation],
		wandb_run: Run | None,
		device: torch.device,
		hookpoints: list[str],
		save_dir: str,
		llms: list = None,
	):
		super().__init__(
			cfg,
			activations_dataloader,
			crosscoder,
			wandb_run,
			device,
			hookpoints,
			save_dir,
		)
		
		# Initialize MLP weights dynamically based on the model
		global W_ins, b_ins, W_outs, b_outs
		if llms and len(llms) > 0:
			llm = llms[0]  # Use first model for MLP weights
			n_layers = llm.cfg.n_layers
			W_ins = torch.stack([llm.blocks[i].mlp.W_in for i in range(n_layers)], dim=0)
			b_ins = torch.stack([llm.blocks[i].mlp.b_in for i in range(n_layers)], dim=0)
			W_outs = torch.stack([llm.blocks[i].mlp.W_out for i in range(n_layers)], dim=0)
			b_outs = torch.stack([llm.blocks[i].mlp.b_out for i in range(n_layers)], dim=0)

		# Extract unique blocks from hookpoints for dynamic processing
		self.unique_blocks = self._extract_blocks_from_hookpoints(hookpoints)
		self.hookpoints_per_block = self._count_hookpoints_per_block(hookpoints, self.unique_blocks)

		# intialise params to track dead features for auxK loss
		self.dead_mask = torch.zeros(crosscoder.hidden_dim, device=device)
		self.num_tokens_since_fired = torch.zeros(crosscoder.hidden_dim, device=device)

		# TODO: get these from config
		self.dead_steps = 1000 # 10M in OAI SAE paper, but we train for far fewer steps
		self.auxK_lambda = cfg.auxK_lambda
		#self.norm_scaling_factors_ML = norm_scaling_factors_ML
		#TODO: add this to config
		self.beta = 5.0
		self.lambda_sharpness=0#100
		#self.llm_model=llm
		#self.tokenizer=tokenizer
		self.W_ins=W_ins
		self.b_ins=b_ins
		self.W_outs=W_outs
		self.b_outs=b_outs
		self.lambda_nsharpness=cfg.lam_n
		self.beta_n_sharpness=cfg.beta_n
		self.sharpness_p=0
		self.mean_max_ratio_preacts=0

	def _extract_blocks_from_hookpoints(self, hookpoints: list[str]) -> list[int]:
		"""Extract unique block numbers from hookpoint names."""
		blocks = []
		for hookpoint in hookpoints:
			if "blocks." in hookpoint:
				# Extract block number (e.g., "blocks.0.hook_resid_pre" -> 0)
				block_num = int(hookpoint.split("blocks.")[1].split(".")[0])
				if block_num not in blocks:
					blocks.append(block_num)
		return sorted(blocks)

	def _count_hookpoints_per_block(self, hookpoints: list[str], unique_blocks: list[int]) -> int:
		"""Count how many hookpoints there are per block."""
		if not unique_blocks:
			return 0
		
		first_block = unique_blocks[0]
		count = 0
		for hookpoint in hookpoints:
			if f"blocks.{first_block}." in hookpoint:
				count += 1
		return count

	# TODO do we need validation toggle?
	def _get_loss(self, batch_BMPD: torch.Tensor, validation=False) -> tuple[torch.Tensor, np.ndarray[Any, np.dtype[np.float64]], torch.Tensor, int]:
		train_res = self.crosscoder.forward_train(batch_BMPD)

		reconstruction_loss = calculate_reconstruction_loss(batch_BMPD, train_res.output_BXD)
		if not validation:
			# create binary mask of dead features
			dead_mask_b = torch.where(train_res.hidden_BH.abs().sum(dim=0) == 0, 1, 0)
			# update counts of tokens since last fired and dead mask
			self.num_tokens_since_fired += dead_mask_b
			self.num_tokens_since_fired[train_res.hidden_BH.abs().sum(dim=0) > 1e-6] = 0
			self.dead_mask = torch.where(self.num_tokens_since_fired > self.dead_steps, 1, 0)
		# calculate auxK loss
		# k_aux = 1/2 residual stream dim
		k_aux = batch_BMPD.shape[-1] // 2
		num_dead = self.dead_mask.sum()
		scale = min(1.0, (num_dead / k_aux).item())
		auxk_latents = torch.where(self.dead_mask == 1, train_res.hidden_BH, torch.zeros_like(train_res.hidden_BH))
		auxk_acts, auxk_indices = auxk_latents.topk(int(k_aux), sorted=False)

		total_variance = (batch_BMPD - batch_BMPD.mean(dim=0)).pow(2).sum()
		e = batch_BMPD - train_res.output_BXD
		# create tensor of zeros
		auxk_act_input = torch.zeros_like(train_res.hidden_BH)
		# fill tensor with topk activations
		auxk_act_input[:, auxk_indices] = auxk_acts


		e_hat = self.crosscoder._decode_BXD(auxk_act_input)
		auxk_loss = (e_hat - e).pow(2).sum()
		auxk_loss = scale * auxk_loss / total_variance

		# TODO: add swish loss here.
		p_PNH_semidata=neuron_sharpness_quick(train_res.hidden_BH,self.crosscoder,self.W_ins,self.b_ins,self.W_outs,self.b_outs,DEVICE)
		neuron_sharpness_semidata=sharpness_func(p_PNH_semidata,self.beta_n_sharpness)
		mean_max_ratio_semidata=(torch.max(p_PNH_semidata,dim=-1)[0]+1e-8)/(p_PNH_semidata.abs().sum(dim=-1)+1e-8)
		mean_max_ratio_semidata=mean_max_ratio_semidata.mean()

		# Extract decoder weights for the blocks used in hookpoints
		# Use block position in unique_blocks list, not the actual block number
		W_dec_PHD=torch.stack([self.crosscoder.W_dec_HXD[:,0,self.hookpoints_per_block*block_pos + (self.hookpoints_per_block-1),:] for block_pos, block in enumerate(self.unique_blocks)],dim=0)
		b_dec_PD=torch.stack([self.crosscoder.b_dec_XD[0,self.hookpoints_per_block*block_pos + (self.hookpoints_per_block-1),:] for block_pos, block in enumerate(self.unique_blocks)],dim=0)

		# Extract MLP weights only for the blocks used in hookpoints
		W_ins_blocks = torch.stack([self.W_ins[block] for block in self.unique_blocks], dim=0)
		b_ins_blocks = torch.stack([self.b_ins[block] for block in self.unique_blocks], dim=0)
		W_outs_blocks = torch.stack([self.W_outs[block] for block in self.unique_blocks], dim=0)
		b_outs_blocks = torch.stack([self.b_outs[block] for block in self.unique_blocks], dim=0)

		# Define penalty functions for block-by-block processing
		def mean_max_ratio_penalty(p_BNH_single):
			"""Compute mean max ratio penalty for a single block."""
			max_f = torch.max(p_BNH_single.abs(), dim=-1)
			sum_f = p_BNH_single.abs().sum(dim=-1)
			ratio = (max_f[0].abs() + 1e-8) / (sum_f + 1e-8)
			return ratio.mean()
		
		def minus_max_penalty(p_BNH_single):
			"""Compute minus max penalty for a single block."""
			max_f = torch.max(p_BNH_single.abs(), dim=-1)
			sum_f = p_BNH_single.abs().sum(dim=-1)
			minus_max = sum_f - max_f[0].abs()
			return minus_max.mean()
		
		# Choose processing approach based on number of blocks
		if len(self.unique_blocks) > 4:
			# Use memory-efficient block-by-block processing for large models
			mean_max_ratio_mlp = add_penalty(
				train_res.hidden_BH, W_dec_PHD, b_dec_PD, W_ins_blocks, b_ins_blocks, 
				W_outs_blocks, b_outs_blocks, DEVICE, bias=1, penalty_fn=mean_max_ratio_penalty
			)
			
			minus_max_mean = add_penalty(
				train_res.hidden_BH, W_dec_PHD, b_dec_PD, W_ins_blocks, b_ins_blocks,
				W_outs_blocks, b_outs_blocks, DEVICE, bias=1, penalty_fn=minus_max_penalty
			)
		else:
			# Use faster all-at-once processing for smaller models (<=4 blocks)
			p_PBNH, _ = get_neuron_preacts_cutoff(
				train_res.hidden_BH, W_dec_PHD, b_dec_PD, W_ins_blocks, b_ins_blocks, 
				W_outs_blocks, b_outs_blocks, DEVICE, bias=1
			)
			
			# Original penalty calculations
			max_f = torch.max(p_PBNH.abs(), dim=-1)
			sum_f = p_PBNH.abs().sum(dim=-1)
			mean_max_ratio_mlp = ((max_f[0].abs() + 1e-8) / (sum_f + 1e-8)).mean()
			
			minus_max = sum_f - max_f[0].abs()
			minus_max_mean = minus_max.mean()
		
		
		

		with torch.no_grad():
			explained_variance_X = calculate_explained_variance_X(batch_BMPD, train_res.output_BXD) # TODO don't bother doing unless logging this batch
			unexplained_variance = []
			for p in range(batch_BMPD.shape[2]):
				unexplained_variance.append(calculate_fvu_X(batch_BMPD[:,:,p,:], train_res.output_BXD[:,:,p,:]).cpu().detach().numpy())
			mean_unexplained_variance = np.mean(np.array(unexplained_variance))

			#add the sharpness func
			sharpness = sharpness_func(train_res.hidden_BH,self.beta)
			#print(f"sharpness: {sharpness}")
			#print(f"mean_max_ratio: {mean_max_ratio}")

			mean_max_ratio_enc=(torch.max(train_res.hidden_BH,dim=-1)[0]+1e-8)/(train_res.hidden_BH.abs().sum(dim=-1)+1e-8)
			mean_max_ratio_enc=mean_max_ratio_enc.mean()

			

			#calculate the neuron pushthrough

		return reconstruction_loss, explained_variance_X.cpu().numpy(), auxk_loss, num_dead, mean_unexplained_variance,sharpness,mean_max_ratio_enc,neuron_sharpness_semidata,mean_max_ratio_semidata,mean_max_ratio_mlp,minus_max_mean#mean_max_ratio_mlp_max,mean_max_ratio_mlp_min
	
	def _train_step(self, batch_BMPD: torch.Tensor) -> dict[str, float]:
		self.optimizer.zero_grad()
		reconstruction_loss, explained_variance_X, auxk_loss, num_dead, mean_unexplained_variance, sharpness, mean_max_ratio_enc,neuron_sharpness_semidata,mean_max_ratio_semidata,mean_max_ratio_mlp,minus_max_mean = self._get_loss(batch_BMPD)#mean_max_ratio_mlp_max,mean_max_ratio_mlp_min
		loss = reconstruction_loss + self.auxK_lambda * auxk_loss+self.lambda_nsharpness*minus_max_mean#+self.lambda_sharpness*sharpness
		loss.backward()
		
		clip_grad_norm_(self.crosscoder.parameters(), 1.0)
		self.optimizer.step()
		assert len(self.optimizer.param_groups) == 1, "sanity check failed"
		self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

		# if self.step%100==0:
		# 	sharps=[]
		# 	mean_max_ratios_preacts=[]
		# 	chunk_size = 1 # You can adjust this parameter as needed

		# 	train_res = self.crosscoder.forward_train(batch_BMPD)
		# 	W_dec_PHD=torch.stack([self.crosscoder.W_dec_HXD[:,0,4*block+3,:] for block in range(4)],dim=0)
		# 	b_dec_PD=torch.stack([self.crosscoder.b_dec_XD[0,4*block+3,:] for block in range(4)],dim=0)
		# 	for batch_start in range(0, train_res.hidden_BH.shape[0], chunk_size):
		# 		# Get the current chunk (handle the case where the last chunk might be smaller)
		# 		batch_end = min(batch_start + chunk_size, train_res.hidden_BH.shape[0])
		# 		current_chunk = train_res.hidden_BH[batch_start:batch_end]
		# 		p_PBNH=get_neuron_preacts(current_chunk,W_dec_PHD,b_dec_PD,self.W_ins,self.b_ins,self.W_outs,self.b_outs,DEVICE)
		# 		sharp=sharpness_func(p_PBNH)
		# 		sharps.append(sharp)
		# 		mean_max_ratio_preacts=(torch.max(p_PBNH,dim=-1)[0]+1e-8)/(p_PBNH.abs().sum(dim=-1)+1e-8)
		# 		mean_max_ratios_preacts.append(mean_max_ratio_preacts.mean())
		# 	self.sharpness_p=torch.tensor(sharps).mean()
		# 	self.mean_max_ratio_preacts=torch.tensor(mean_max_ratios_preacts).mean()


		# TODO should we not be logging running averages of these?
		if (
			self.wandb_run is not None
			and self.cfg.log_every_n_steps is not None
			and (self.step + 1) % self.cfg.log_every_n_steps == 0
		):
			#print(f"step: {self.step}, sharpness_p: {sharpness_p:.0f}, nsharpness loss: {self.lambda_nsharpness*sharpness_p:.0f},rec_loss: {reconstruction_loss.item():.0f}")

			log_dict = {
				"train/reconstruction_loss": reconstruction_loss.item(),
				"train/mean_explained_variance": explained_variance_X.mean(), # TODO check this is right with X instead of ML
				"train/mean_unexplained_variance": mean_unexplained_variance,
				"train/lr": self.optimizer.param_groups[0]["lr"],
				"train/unscaled_auxk_loss": auxk_loss.item(),
				"train/frac_dead_fts": num_dead / self.crosscoder.hidden_dim,
				"train/auxk_loss": self.auxK_lambda * auxk_loss.item(),
				"train/sharpness": sharpness,
				#"train/sharpness_p": self.sharpness_p,
				"train/mean_max_ratio_enc": mean_max_ratio_enc,
				#"train/mean_max_ratio_preacts": self.mean_max_ratio_preacts,
				"train/mean_max_ratio_semidata": mean_max_ratio_semidata,
				"train/neuron_sharpness_semidata": neuron_sharpness_semidata,
				#"train/sharpness_loss": self.lambda_sharpness*sharpness,
				#"train/nsharpness_loss": self.lambda_nsharpness*neuron_sharpness_mlp,
				#"train/nsharpness_loss": self.lambda_nsharpness*neuron_sharpness_mlp,
				#"train/neuron_sharpness_mlp": neuron_sharpness_mlp,
				"train/minus_max_mean":minus_max_mean,
				"train/minus_max_mean_loss":self.lambda_nsharpness*minus_max_mean,
				"train/mean_max_ratio_mlp": mean_max_ratio_mlp,
				"train/xc_dec_norm":self.crosscoder.W_dec_HXD.norm(),
				#"train/mean_max_ratio_mlp_max": mean_max_ratio_mlp_max,
				#"train/mean_max_ratio_mlp_min": mean_max_ratio_mlp_min,
			}

			self.wandb_run.log(log_dict, step=self.step)