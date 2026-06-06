import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from typing import Any
from model_diffing.models.activations.jumprelu import JumpReLUActivation
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from sleepers.scripts.train_jan_update_sleeper.config import JanUpdateTrainConfig
from model_diffing.utils import (
	calculate_explained_variance_X,
	calculate_reconstruction_loss,
	get_decoder_norms_H,
	l0_norm,
)
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from sleepers.scripts.utils import neuron_sharpness_quick,sharpness_func,calculate_fvu_X,get_neuron_preacts

from sleepers.scripts.llms import build_llm_lora

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm = build_llm_lora(
	base_model_repo="roneneldan/TinyStories-Instruct-33M",
	lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
	cache_dir=None,
	device=DEVICE,
	dtype=None
	)
tokenizer = llm.tokenizer

W_ins=torch.stack([llm.blocks[val].mlp.W_in for val in range(4)],dim=0)
b_ins=torch.stack([llm.blocks[val].mlp.b_in for val in range(4)],dim=0)
W_outs=torch.stack([llm.blocks[val].mlp.W_out for val in range(4)],dim=0)
b_outs=torch.stack([llm.blocks[val].mlp.b_out for val in range(4)],dim=0)


class JanUpdateSleeperTrainer(BaseModelHookpointTrainer[JanUpdateTrainConfig, JumpReLUActivation]):
	def _get_loss(self, batch_BMPD: torch.Tensor
				  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
							 np.ndarray[Any, np.dtype[np.float64]]]:
		train_res = self.crosscoder.forward_train(batch_BMPD)

		reconstruction_loss = calculate_reconstruction_loss(batch_BMPD, train_res.output_BXD)
		decoder_norms_H = get_decoder_norms_H(self.crosscoder.W_dec_HXD)
		tanh_sparsity_loss = self._tanh_sparsity_loss(train_res.hidden_BH, decoder_norms_H)
		pre_act_loss = self._pre_act_loss(train_res.hidden_BH, decoder_norms_H)

		p_PNH_semidata=neuron_sharpness_quick(train_res.hidden_BH,self.crosscoder,W_ins,b_ins,W_outs,b_outs,DEVICE)
		#neuron_sharpness_semidata=sharpness_func(p_PNH_semidata,self.cfg.beta_n)
		mean_max_ratio_semidata=(torch.max(p_PNH_semidata,dim=-1)[0]+1e-8)/(p_PNH_semidata.abs().sum(dim=-1)+1e-8)
		mean_max_ratio_semidata=mean_max_ratio_semidata.mean()
		
		#print(f'lambda n: {self.cfg.lam_n}, beta n: {self.cfg.beta_n}')
		
		lambda_s = self._lambda_s_scheduler()
		scaled_tanh_sparsity_loss = lambda_s * tanh_sparsity_loss
		scaled_pre_act_loss = self.cfg.lambda_p * pre_act_loss
		#scaled_semidata_loss = self.cfg.lam_n * mean_max_ratio_semidata

		loss = reconstruction_loss + scaled_tanh_sparsity_loss + scaled_pre_act_loss#+scaled_semidata_loss
		mean_l0 = l0_norm(train_res.hidden_BH, dim=-1).mean()

		
		#explained_variance_X = calculate_explained_variance_X(batch_BMPD, train_res.output_BXD)

		with torch.no_grad():
			
			unexplained_variance = []
			for p in range(batch_BMPD.shape[2]):
				unexplained_variance.append(calculate_fvu_X(batch_BMPD[:,:,p,:], train_res.output_BXD[:,:,p,:]).cpu().detach().numpy())
			mean_unexplained_variance = np.mean(np.array(unexplained_variance))

		return (loss, reconstruction_loss, mean_l0, tanh_sparsity_loss, pre_act_loss,mean_max_ratio_semidata,
				mean_unexplained_variance)


	def _train_step(self, batch_BMPD: torch.Tensor) -> None:
		self.optimizer.zero_grad()

		(loss, reconstruction_loss, mean_l0, tanh_sparsity_loss, pre_act_loss,mean_max_ratio_semidata,
				unexplained_variance_X) = self._get_loss(batch_BMPD)
		loss.backward()
		clip_grad_norm_(self.crosscoder.parameters(), 1.0)
		self.optimizer.step()
		self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)
		lambda_s = self._lambda_s_scheduler()

		if (
			self.wandb_run is not None
			and self.cfg.log_every_n_steps is not None
			and (self.step + 1) % self.cfg.log_every_n_steps == 0
		):
			log_dict = {
				"train/reconstruction_loss": reconstruction_loss.item(),
				"train/mean_l0": mean_l0.item(),    
				"train/mean_l0_pct": mean_l0.item() / self.crosscoder.hidden_dim,
				"train/reconstruction_loss": reconstruction_loss.item(),
				"train/unscaled_tanh_sparsity_loss": tanh_sparsity_loss.item(),
				"train/scaled_tanh_sparsity_loss": lambda_s * tanh_sparsity_loss.item() / (lambda_s+1e-8),
				"train/unscaled_pre_act_loss": pre_act_loss.item(),
				"train/scaled_pre_act_loss": self.cfg.lambda_p * pre_act_loss.item(),
				"train/unexplained_variance": unexplained_variance_X.item(),
				"train/lr": self.optimizer.param_groups[0]["lr"],
				"train/mean_max_ratio_semidata": mean_max_ratio_semidata.item(),
			}

			self.wandb_run.log(log_dict, step=self.step)

	def _lambda_s_scheduler(self) -> float:
		"""linear ramp from 0 to lambda_s over the course of training"""
		return (self.cfg.initial_lambda_s + 
				(self.step / self.total_steps) * 
				(self.cfg.final_lambda_s - self.cfg.initial_lambda_s))

	def _tanh_sparsity_loss(self, hidden_BH: torch.Tensor, decoder_norms_H: torch.Tensor) -> torch.Tensor:
		loss_BH = torch.tanh(self.cfg.c * hidden_BH * decoder_norms_H)
		return loss_BH.sum(-1).mean()

	def _pre_act_loss(self, hidden_BH: torch.Tensor, decoder_norms_H: torch.Tensor) -> torch.Tensor:
		t_H = self.crosscoder.hidden_activation.log_threshold_H
		loss_BH = torch.relu(t_H.exp() - hidden_BH) * decoder_norms_H
		return loss_BH.sum(-1).mean()
