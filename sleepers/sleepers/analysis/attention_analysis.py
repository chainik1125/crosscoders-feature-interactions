"""
Attention analysis module for crosscoder features.

This module provides functionality to analyze attention patterns in language models
with crosscoder features, including:
- Attention pattern computation and visualization
- Feature interaction analysis
- MLP neuron prediction analysis
- Attention decomposition into feature contributions
"""

from code import interact
import logging
import numpy as np
from sympy.logic import false
import torch
from einops import einsum, rearrange
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.cm as cm
from tqdm import tqdm
import yaml
import sys
import os
from datetime import datetime
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sleepers.scripts.llms import build_llm_lora
from datasets import load_dataset
from sleepers.scripts.utils import load_crosscoder_from_wandb
import pandas as pd

from collections import defaultdict



def setup_logger(name: str = "attention_analysis", level: str = "INFO", 
				log_file: Optional[str] = None) -> logging.Logger:
	"""
	Set up a logger with configurable level and output.
	
	Args:
		name: Logger name
		level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
		log_file: Optional file path to log to
		
	Returns:
		Configured logger instance
	"""
	logger = logging.getLogger(name)
	logger.setLevel(getattr(logging, level.upper()))
	
	# Clear existing handlers
	logger.handlers.clear()
	
	# Create formatter
	formatter = logging.Formatter(
		'%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S'
	)
	
	# Console handler
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setLevel(getattr(logging, level.upper()))
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)
	
	# File handler if specified
	if log_file:
		file_handler = logging.FileHandler(log_file)
		file_handler.setLevel(getattr(logging, level.upper()))
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
	
	return logger


def load_config(config_path: str) -> Dict[str, Any]:
	"""
	Load configuration from YAML file.
	
	Args:
		config_path: Path to YAML configuration file
		
	Returns:
		Configuration dictionary
	"""
	try:
		with open(config_path, 'r') as f:
			config = yaml.safe_load(f)
		return config if config else {}
	except FileNotFoundError:
		logging.warning(f"Config file not found: {config_path}")
		return {}
	except yaml.YAMLError as e:
		logging.error(f"Error parsing YAML config: {e}")
		return {}


class AttentionAnalyzer:
	"""
	Main class for analyzing attention patterns in crosscoder models.
	"""
	
	def __init__(self, crosscoder, llm, tokenizer, dataloader_mean, hookpoints, 
				 device='cpu', logger=None, config_path=None):
		"""
		Initialize the attention analyzer.
		
		Args:
			crosscoder: The crosscoder model
			llm: The language model
			tokenizer: The tokenizer
			dataloader_mean: Mean activations from dataloader
			hookpoints: List of hookpoint names
			device: Device to run computations on
			logger: Logger instance (optional)
			config_path: Path to YAML configuration file (optional)
		"""
		self.crosscoder = crosscoder
		self.llm = llm
		self.tokenizer = tokenizer
		self.dataloader_mean = dataloader_mean
		self.hookpoints = hookpoints
		self.device = device
		
		# Load configuration first
		self.config = {}
		if config_path:
			self.config = load_config(config_path)
		else:
			# Try to load default config
			default_config_path = Path(__file__).parent / "attention_analysis.yaml"
			if default_config_path.exists():
				self.config = load_config(str(default_config_path))
		
		# Set up logger with config
		if logger is None:
			log_config = self.config.get('logging', {})
			self.logger = setup_logger(
				"attention_analysis", 
				level=log_config.get('level', 'INFO'),
				log_file=log_config.get('file')
			)
		else:
			self.logger = logger
		
		# Log configuration loading
		if config_path:
			self.logger.info(f"Loaded configuration from {config_path}")
		elif (Path(__file__).parent / "attention_analysis.yaml").exists():
			self.logger.info(f"Loaded default configuration")
		
		# Apply configuration settings
		self._apply_config()
		
		# Pre-compute feature activation statistics
		self.feature_activations_mean = None
		self.feature_activations_active_mean = None
		self.feature_activations_active_count = None
		
		self.logger.info(f"Initialized AttentionAnalyzer with device: {device}")
		self.logger.info(f"Model has {len(hookpoints)} hookpoints")
		self.logger.info(f"Crosscoder hidden dimension: {crosscoder.hidden_dim}")
		
	def _apply_config(self):
		"""Apply configuration settings to class attributes."""
		# Set default values from config
		self.max_sequence_length = self.config.get('model', {}).get('max_sequence_length', 128)
		self.plot_size = self.config.get('analysis', {}).get('attention', {}).get('plot_size', [10, 8])
		self.colormap = self.config.get('analysis', {}).get('attention', {}).get('colormap', 'viridis')
		self.show_tokens = self.config.get('analysis', {}).get('attention', {}).get('show_tokens', 30)
		self.dpi = self.config.get('analysis', {}).get('visualization', {}).get('dpi', 160)
		self.figure_size = self.config.get('analysis', {}).get('visualization', {}).get('figure_size', [12, 6])
		
		# Set matplotlib DPI if configured
		if self.dpi:
			plt.rcParams['figure.dpi'] = self.dpi
			
		self.logger.debug(f"Applied configuration: max_seq_len={self.max_sequence_length}, "
						 f"plot_size={self.plot_size}, colormap={self.colormap}")
		
	def get_llm_activations(self, input_text: str, subtract_mean: bool = True) -> torch.Tensor:
		"""
		Get LLM activations for given input text.
		
		Args:
			input_text: Input text to analyze
			subtract_mean: Whether to subtract the dataloader mean
			
		Returns:
			Tensor of shape (sequence_length, 1, num_hookpoints, hidden_dim)
		"""
		self.logger.debug(f"Getting LLM activations for text of length {len(input_text)}")
		tokens = torch.tensor(self.tokenizer.encode(input_text)[:self.max_sequence_length])
		self.logger.debug(f"Tokenized to {len(tokens)} tokens")
		_, cache = self.llm.run_with_cache(tokens.unsqueeze(0), names_filter=self.hookpoints)
		
		activations_BSPD = torch.stack([cache[name] for name in cache.keys()], dim=2)
		if subtract_mean:
			activations_BSPD -= self.dataloader_mean[:tokens.shape[0], 0, :, :].unsqueeze(0)
		
		activations_BSMPD = torch.unsqueeze(activations_BSPD, dim=2)
		activations_SMPD = rearrange(activations_BSMPD, "b s m l d -> (b s) m l d")
		return activations_SMPD

	def attention_pattern_QK(self, layer: int, head: int, q_input: torch.Tensor, 
						   q_do_bias: bool, k_input: torch.Tensor, k_do_bias: bool) -> np.ndarray:
		"""
		Compute attention pattern from query and key inputs.
		
		Args:
			layer: Layer index
			head: Head index
			q_input: Query input tensor
			q_do_bias: Whether to add query bias
			k_input: Key input tensor
			k_do_bias: Whether to add key bias
			
		Returns:
			Attention scores as numpy array
		"""
		W_Q = self.llm.blocks[layer].attn.W_Q[head]
		b_Q = self.llm.blocks[layer].attn.b_Q[head]
		W_K = self.llm.blocks[layer].attn.W_K[head]
		b_K = self.llm.blocks[layer].attn.b_K[head]
		
		q = einsum(W_Q, q_input, "d a, s d -> s a")
		if q_do_bias:
			q += b_Q
			
		k = einsum(W_K, k_input, "d a, s d -> s a")
		if k_do_bias:
			k += b_K
			
		attention_scores = einsum(q, k, "q a, k a -> q k")
		return attention_scores.to("cpu").numpy()

	def lower_triangular_mask(self, pattern: np.ndarray) -> np.ma.MaskedArray:
		"""Apply lower triangular mask to attention pattern."""
		mask = np.triu(np.ones(pattern.shape), k=1)
		return np.ma.array(np.tril(pattern, k=0), mask=mask)

	def softmax_pattern(self, attention_pattern: np.ndarray) -> np.ma.MaskedArray:
		"""Apply softmax to attention pattern with causal masking."""
		attention_pattern = np.tril(attention_pattern)
		softmaxed_attn_scores = np.zeros_like(attention_pattern)
		
		for q in range(attention_pattern.shape[0]):
			attn = attention_pattern[q, :q+1]
			if len(attn) > 0:
				attn_max = np.max(attn)
				attn_exp = np.exp(attn - attn_max)
				attn_softmax = attn_exp / np.sum(attn_exp)
				softmaxed_attn_scores[q, :q+1] = attn_softmax
				
		return self.lower_triangular_mask(softmaxed_attn_scores)

	def pattern_subtract_row_mean(self, pattern: np.ma.MaskedArray) -> np.ma.MaskedArray:
		"""Subtract row mean from attention pattern."""
		mean_subtracted_pattern = np.zeros_like(pattern)
		for q in range(pattern.shape[0]):
			if q > 0:
				row_mean = np.mean(pattern[q, :q+1])
				mean_subtracted_pattern[q, :q+1] = pattern[q, :q+1] - row_mean
		return mean_subtracted_pattern

	def get_decomposed_attention_pattern(self, layer: int, head: int, q_do_mean: bool, 
									   k_do_mean: bool, input_text: str) -> np.ndarray:
		"""
		Get decomposed attention pattern for specific layer/head.
		
		Args:
			layer: Layer index
			head: Head index
			q_do_mean: Whether to use mean for query
			k_do_mean: Whether to use mean for key
			input_text: Input text
			
		Returns:
			Decomposed attention scores
		"""
		self.logger.debug(f"Computing decomposed attention pattern for layer {layer}, head {head}")
		self.logger.debug(f"Query mean: {q_do_mean}, Key mean: {k_do_mean}")
		tokens = torch.tensor(self.tokenizer.encode(input_text)[:self.max_sequence_length])
		_, cache = self.llm.run_with_cache(tokens.unsqueeze(0), names_filter=[
			f"blocks.{layer}.ln1.hook_normalized",
		])
		
		activations_SD = cache[f"blocks.{layer}.ln1.hook_normalized"][0]
		mean_input_SD = self.dataloader_mean[:, 0, self.hookpoints.index(f"blocks.{layer}.ln1.hook_normalized"), :]
		
		k_input_KD = mean_input_SD if k_do_mean else activations_SD - mean_input_SD
		q_input_QD = mean_input_SD if q_do_mean else activations_SD - mean_input_SD
		
		attention_scores = self.attention_pattern_QK(layer, head, q_input_QD, q_do_mean, k_input_KD, k_do_mean)
		attention_scores = self.pattern_subtract_row_mean(self.lower_triangular_mask(attention_scores))
		
		return attention_scores

	def plot_attention_pattern(self, attention_scores: np.ndarray, title: str, 
							 xlabel: str = 'Key position', ylabel: str = 'Query position'):
		"""Plot attention pattern with proper visualization."""
		cmap = getattr(plt.cm, self.colormap).copy()
		color_bad = self.config.get('analysis', {}).get('visualization', {}).get('color_bad', 'black')
		cmap.set_bad(color_bad)
		
		plt.figure(figsize=self.plot_size)
		plt.imshow(attention_scores, cmap=cmap, interpolation='none')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.colorbar()
		plt.xticks(np.arange(0, attention_scores.shape[1], 10))
		plt.yticks(np.arange(0, attention_scores.shape[0], 10))
		plt.show()

	def compute_feature_activation_statistics(self, dataset, n_datapoints: int = 100):
		"""
		Compute statistics for feature activations across dataset.
		
		Args:
			dataset: Dataset to compute statistics on
			n_datapoints: Number of datapoints to use
		"""
		self.logger.info(f"Computing feature activation statistics over {n_datapoints} datapoints")
		hidden_dim = self.crosscoder.hidden_dim
		
		self.feature_activations_mean = np.zeros(hidden_dim)
		self.feature_activations_active_mean = np.zeros(hidden_dim)
		self.feature_activations_active_count = np.zeros(hidden_dim)
		
		for i in range(n_datapoints):
			activations_SMPD = self.get_llm_activations(dataset[i]["text"], subtract_mean=True)
			feature_activations_BH = self.crosscoder._encode_BH(activations_SMPD).to("cpu").numpy()
			
			self.feature_activations_mean += np.mean(feature_activations_BH, axis=0)
			self.feature_activations_active_mean += np.sum(feature_activations_BH, axis=0)
			self.feature_activations_active_count += np.sum(feature_activations_BH != 0, axis=0)
			
		self.feature_activations_mean /= n_datapoints
		
		for i in range(hidden_dim):
			if self.feature_activations_active_count[i] > 0:
				self.feature_activations_active_mean[i] /= self.feature_activations_active_count[i]
			else:
				self.feature_activations_active_mean[i] = 0
		
		active_features = np.sum(self.feature_activations_active_count > 0)
		self.logger.info(f"Computed statistics: {active_features}/{hidden_dim} features are active")
		self.logger.info(f"Mean activation when active: {np.mean(self.feature_activations_active_mean):.4f}")

	def analyze_attention_decomposition(self, input_text: str, layer: int, head: int, 
									  query_pos: int, show_tokens: int = None):
		"""
		Analyze attention decomposition into different components.
		
		Args:
			input_text: Input text to analyze
			layer: Layer index
			head: Head index
			query_pos: Query position to analyze
			show_tokens: Number of tokens to show in plot
		"""
		if show_tokens is None:
			show_tokens = self.show_tokens
			
		self.logger.info(f"Analyzing attention decomposition for layer {layer}, head {head}, query pos {query_pos}")
		general_attention = self.get_decomposed_attention_pattern(layer, head, True, True, input_text)
		key_attention = self.get_decomposed_attention_pattern(layer, head, True, False, input_text)
		query_attention = self.get_decomposed_attention_pattern(layer, head, False, True, input_text)
		key_query_attention = self.get_decomposed_attention_pattern(layer, head, False, False, input_text)
		total_attention = general_attention + key_attention + query_attention + key_query_attention
		
		tokens = self.tokenizer.encode(input_text)[:self.max_sequence_length]
		token_strings = [self.tokenizer.decode([token]) for token in tokens[:query_pos+1]]
		
		plt.figure(figsize=self.figure_size)
		start_pos = max(0, query_pos - show_tokens + 1)
		end_pos = query_pos + 1
		
		plt.plot(general_attention[query_pos, start_pos:end_pos], label='General attention')
		plt.plot(key_attention[query_pos, start_pos:end_pos], label='Key attention')
		plt.plot(query_attention[query_pos, start_pos:end_pos], label='Query attention')
		plt.plot(key_query_attention[query_pos, start_pos:end_pos], label='Key query attention')
		plt.plot(total_attention[query_pos, start_pos:end_pos], label='Total attention')
		
		plt.xticks(range(len(token_strings[start_pos:end_pos])), 
				  token_strings[start_pos:end_pos], rotation=45, ha='right')
		plt.xlabel('Token Position')
		plt.ylabel('Attention Score')
		plt.title(f'Attention Patterns for Layer {layer} Head {head} at Query Position {query_pos}')
		plt.tight_layout()
		plt.legend()
		plt.show()

	def analyze_feature_mlp_interactions(self, input_text: str, layer: int, sequence_pos: int):
		"""
		Analyze how features interact with MLP neurons.
		
		Args:
			input_text: Input text to analyze
			layer: Layer index
			sequence_pos: Sequence position to analyze
		"""
		tokens = torch.tensor(self.tokenizer.encode(input_text)[:self.max_sequence_length])
		_, cache = self.llm.run_with_cache(tokens.unsqueeze(0), 
										  names_filter=self.hookpoints + [f"blocks.{layer}.hook_mlp_out"])
		
		activations_SMPD = self.get_llm_activations(input_text, subtract_mean=True)
		feature_activations_SH = self.crosscoder._encode_BH(activations_SMPD)
		
		decode_SMPD = self.crosscoder._decode_BXD(feature_activations_SH) + \
					 self.dataloader_mean[:tokens.shape[0], 0, :, :].unsqueeze(1)
		
		decode_mlp_input_SD = decode_SMPD[:, 0, self.hookpoints.index(f'blocks.{layer}.ln2.hook_normalized'), :]
		
		W_in = self.llm.blocks[layer].mlp.W_in
		b_in = self.llm.blocks[layer].mlp.b_in
		
		decode_mlp_preact_SU = decode_mlp_input_SD @ W_in + b_in
		
		mean_at_mlp_input_SU = self.dataloader_mean[:, 0, self.hookpoints.index(f"blocks.{layer}.ln2.hook_normalized"), :] @ W_in + b_in
		features_at_mlp_input_HU = self.crosscoder.W_dec_HXD[:, 0, self.hookpoints.index(f"blocks.{layer}.ln2.hook_normalized"), :] @ W_in
		
		if self.feature_activations_active_mean is not None:
			features_at_mlp_input_HU = features_at_mlp_input_HU.to("cpu").numpy() * self.feature_activations_active_mean[:, np.newaxis]
		
		return {
			'decode_mlp_preact': decode_mlp_preact_SU,
			'mean_at_mlp_input': mean_at_mlp_input_SU,
			'features_at_mlp_input': features_at_mlp_input_HU,
			'feature_activations': feature_activations_SH
		}

	def analyze_feature_firing_patterns(self, dataset, layer: int, neuron: int, n_examples: int = 1000):
		"""
		Analyze firing patterns of features with respect to MLP neurons.
		
		Args:
			dataset: Dataset to analyze
			layer: Layer index
			neuron: Neuron index
			n_examples: Number of examples to analyze
		"""
		self.logger.info(f"Analyzing feature firing patterns for layer {layer}, neuron {neuron} over {n_examples} examples")
		feature_positive_seen = np.zeros(self.crosscoder.hidden_dim)
		feature_negative_seen = np.zeros(self.crosscoder.hidden_dim)
		
		W_in = self.llm.blocks[layer].mlp.W_in
		b_in = self.llm.blocks[layer].mlp.b_in
		
		for i in tqdm(range(n_examples), desc="Computing feature firing patterns"):
			input_text = dataset[i]["text"]
			
			activations_SMPD = self.get_llm_activations(input_text, subtract_mean=True)
			feature_activations_SH = self.crosscoder._encode_BH(activations_SMPD)
			
			decode_SMPD = self.crosscoder._decode_BXD(feature_activations_SH) + \
						 self.dataloader_mean[:activations_SMPD.shape[0], :, :, :]
			
			decode_mlp_input_SD = decode_SMPD[:, 0, self.hookpoints.index(f'blocks.{layer}.ln2.hook_normalized'), :]
			mlp_preactivations_SU = decode_mlp_input_SD @ W_in + b_in
			
			for sequence_pos in range(activations_SMPD.shape[0]):
				active_features = torch.where(feature_activations_SH[sequence_pos] != 0)[0].to("cpu").numpy()
				preact = mlp_preactivations_SU[sequence_pos, neuron]
				
				if preact > 0:
					feature_positive_seen[active_features] += 1
				else:
					feature_negative_seen[active_features] += 1
		
		total_positive = np.sum(feature_positive_seen)
		total_negative = np.sum(feature_negative_seen)
		self.logger.info(f"Analysis complete: {total_positive} positive, {total_negative} negative feature activations")
		return feature_positive_seen, feature_negative_seen

	def analyze_feature_attention_interactions(self, layer: int, head: int, 
											 input_text: str, query_position: int, key_position: int):
		"""
		Analyze interactions between features in attention.
		
		Args:
			layer: Layer index
			head: Head index
			input_text: Input text to analyze
			query_position: Query position
			key_position: Key position
		"""
		activations_SMPD = self.get_llm_activations(input_text, subtract_mean=True)
		feature_activations_SH = self.crosscoder._encode_BH(activations_SMPD)
		
		feature_activations_query = feature_activations_SH[query_position]
		query_active_features = torch.where(feature_activations_query != 0)[0]
		
		feature_activations_key = feature_activations_SH[key_position]
		key_active_features = torch.where(feature_activations_key != 0)[0]
		
		query_activations_for_features = self.crosscoder.W_dec_HXD[
			query_active_features, 0, self.hookpoints.index(f"blocks.{layer}.ln1.hook_normalized")
		]
		key_activations_for_features = self.crosscoder.W_dec_HXD[
			key_active_features, 0, self.hookpoints.index(f"blocks.{layer}.ln1.hook_normalized")
		]
		
		interaction_matrix_unscaled = self.attention_pattern_QK(
			layer, head,
			query_activations_for_features, False,
			key_activations_for_features, False
		)
		
		query_active_features = query_active_features.cpu().numpy()
		key_active_features = key_active_features.cpu().numpy()
		
		matrix_scaling = feature_activations_query[query_active_features].unsqueeze(1) * \
						feature_activations_key[key_active_features].unsqueeze(0)
		matrix_scaling = matrix_scaling.to("cpu").numpy()
		
		if self.feature_activations_active_mean is not None:
			interaction_matrix_unscaled *= self.feature_activations_active_mean[query_active_features][:, np.newaxis]
			interaction_matrix_unscaled *= self.feature_activations_active_mean[key_active_features][np.newaxis, :]
			matrix_scaling /= self.feature_activations_active_mean[query_active_features][:, np.newaxis]
			matrix_scaling /= self.feature_activations_active_mean[key_active_features][np.newaxis, :]
		
		return {
			'query_active_features': query_active_features,
			'key_active_features': key_active_features,
			'interaction_matrix_unscaled': interaction_matrix_unscaled,
			'matrix_scaling': matrix_scaling,
			'interaction_matrix': interaction_matrix_unscaled * matrix_scaling
		}

	def data_independent_attention(self, layer: int, head: int):
		"""
		Analyze interactions between features in attention.
		
		Args:
			layer: Layer index
			head: Head index
			input_text: Input text to analyze
			query_position: Query position
			key_position: Key position
		"""
		
		
		
		query_activations_for_features = self.crosscoder.W_dec_HXD[
			:, 0, self.hookpoints.index(f"blocks.{layer}.ln1.hook_normalized")
		]
		key_activations_for_features = self.crosscoder.W_dec_HXD[
			:, 0, self.hookpoints.index(f"blocks.{layer}.ln1.hook_normalized")
		]
		
		interaction_matrix_unscaled = self.attention_pattern_QK(
			layer, head,
			query_activations_for_features, False,
			key_activations_for_features, False
		)
		
		
		return interaction_matrix_unscaled
	
	def feature_interactions_whole_text(self,key_feat_index:int,query_feat_index:int,input_text:str,layer:int,head:int):
		
		activations_SMPD = self.get_llm_activations(input_text, subtract_mean=True)
		feature_activations_SH = self.crosscoder._encode_BH(activations_SMPD)

		
		#feature_activations_query = feature_activations_SH[query_position]
		#query_active_features = torch.where(feature_activations_query != 0)[0]
		
		#feature_activations_key = feature_activations_SH[key_position]
		#key_active_features = torch.where(feature_activations_key != 0)[0]
		
		query_activations_for_features = self.crosscoder.W_dec_HXD[
			query_feat_index, 0, self.hookpoints.index(f"blocks.{layer}.ln1.hook_normalized")
		].unsqueeze(0)
		key_activations_for_features = self.crosscoder.W_dec_HXD[
			key_feat_index, 0, self.hookpoints.index(f"blocks.{layer}.ln1.hook_normalized")
		].unsqueeze(0)
		
		interaction_matrix_unscaled = self.attention_pattern_QK(
			layer, head,
			query_activations_for_features, False,
			key_activations_for_features, False
		)
		
		#query_active_features = query_active_features.cpu().numpy()
		#key_active_features = key_active_features.cpu().numpy()
		
			
		matrix_scaling = feature_activations_SH[:,query_feat_index].unsqueeze(1) * \
						feature_activations_SH[:,key_feat_index].unsqueeze(0).clamp(min=1e-4)
						
		matrix_scaling = matrix_scaling.to("cpu").numpy()
		
		if self.feature_activations_active_mean is not None:
			print(f'mean triggered, shape: {self.feature_activations_active_mean.shape}')
			interaction_matrix_unscaled *= self.feature_activations_active_mean[query_feat_index][:, np.newaxis]
			interaction_matrix_unscaled *= self.feature_activations_active_mean[key_feat_index][np.newaxis, :]
			matrix_scaling /= self.feature_activations_active_mean[query_feat_index][:, np.newaxis]
			matrix_scaling /= self.feature_activations_active_mean[key_feat_index][np.newaxis, :]
		
		return {
			'query_feature': query_feat_index,
			'key_feature': key_feat_index,
			'interaction_matrix_unscaled': interaction_matrix_unscaled,
			'matrix_scaling': matrix_scaling,
			'interaction_matrix': interaction_matrix_unscaled * matrix_scaling
		}


class FeatureAnalyzer:
	"""
	Utility class for analyzing individual features.
	"""
	
	def __init__(self, crosscoder, tokenizer, logger=None):
		self.crosscoder = crosscoder
		self.tokenizer = tokenizer
		
		# Set up logger
		if logger is None:
			self.logger = setup_logger("feature_analysis")
		else:
			self.logger = logger
			
		self.logger.info(f"Initialized FeatureAnalyzer with {crosscoder.hidden_dim} features")
		
	def get_top_activations(self, dataset, hidden_dim: int, N: int = 5, n_prompts: int = 100):
		"""
		Get top N activations for each feature across the dataset.
		
		Args:
			dataset: Dataset to analyze
			hidden_dim: Hidden dimension size
			N: Number of top activations to keep
			n_prompts: Number of prompts to analyze
			
		Returns:
			List of lists containing top activations for each feature
		"""
		from itertools import islice
		
		self.logger.info(f"Finding top {N} activations for {hidden_dim} features across {n_prompts} prompts")
		top_activations = [[] for _ in range(hidden_dim)]
		
		for example in islice(dataset, n_prompts):
			activations_SMPD = self.get_llm_activations(example["text"], subtract_mean=True)
			feature_activations_SH = self.crosscoder._encode_BH(activations_SMPD)
			
			for seq_pos in range(feature_activations_SH.shape[0]):
				active_features = torch.nonzero(feature_activations_SH[seq_pos]).squeeze()
				
				if active_features.ndim == 0:
					active_features = [active_features.item()]
					
				for feature_idx in active_features:
					activation_val = feature_activations_SH[seq_pos, feature_idx].item()
					top_activations[feature_idx].append((activation_val, example, seq_pos))
					
					top_activations[feature_idx].sort(key=lambda x: x[0], reverse=True)
					if len(top_activations[feature_idx]) > N:
						top_activations[feature_idx] = top_activations[feature_idx][:N]
		
		active_features = sum(1 for activations in top_activations if len(activations) > 0)
		self.logger.info(f"Found activations for {active_features}/{hidden_dim} features")
		return top_activations
	
	def format_token_in_example(self, top_activation):
		"""Format a token with surrounding context."""
		if top_activation is None:
			return "None"
			
		example = top_activation[1]["text"]
		token_index = top_activation[2]
		tokens = self.tokenizer.encode(example)
		
		open_bracket = self.tokenizer.encode("[")[0]
		close_bracket = self.tokenizer.encode("]")[0]
		
		tokens.insert(token_index, open_bracket)
		tokens.insert(token_index + 2, close_bracket)
		start = max(0, token_index - 2)
		return self.tokenizer.decode(tokens[start:token_index + 4])
	
	def format_tokens_in_examples(self, top_activations):
		"""Format multiple tokens with context."""
		return [self.format_token_in_example(top_activation) for top_activation in top_activations]


def create_interactive_attention_visualizer(tokens, cache_attn_scores, tokenizer, layer, head):
	"""
	Create an interactive attention visualizer using ipywidgets.
	
	Args:
		tokens: Token tensor
		cache_attn_scores: Cached attention scores
		tokenizer: Tokenizer
		layer: Layer index
		head: Head index
	"""
	try:
		import ipywidgets as widgets
		from IPython.display import display, HTML
		
		def lower_triangular_mask(pattern):
			mask = np.triu(np.ones(pattern.shape), k=1)
			return np.ma.array(np.tril(pattern, k=0), mask=mask)
		
		cache_attn_scores = lower_triangular_mask(cache_attn_scores.to("cpu").numpy())
		
		def update_display(q_idx, tokens):
			with output_area:
				output_area.clear_output(wait=True)
				
				attn_scores = cache_attn_scores[q_idx, :q_idx+1]
				attn_scores_softmax = np.exp(attn_scores) / np.sum(np.exp(attn_scores))
				
				if attn_scores_softmax.size > 0:
					min_val = attn_scores_softmax.min()
					max_val = attn_scores_softmax.max()
					range_val = max_val - min_val if max_val - min_val != 0 else 1.0
					norm_attn = (attn_scores_softmax - min_val) / range_val
				else:
					norm_attn = attn_scores_softmax
				
				html_str = f"<div style='margin-bottom:8px;'><b>Layer {layer} head {head} attention for token {q_idx}</b></div>"
				for i, token in enumerate(tokens):
					if i <= q_idx:
						alpha = norm_attn[i] if i < len(norm_attn) else 0.0
						
						if i == q_idx:
							style = f"background-color: rgba(255, 0, 0, {alpha}); padding:2px; margin:2px; border: 3px solid blue; border-radius: 5px;"
						else:
							style = f"background-color: rgba(255, 0, 0, {alpha}); padding:2px; margin:2px;"
					else:
						style = "padding:2px; margin:2px;"
					html_str += f"<span style='{style}'>{tokenizer.decode([token])}</span> "
				
				display(HTML(html_str))
				
				fig, ax = plt.subplots(figsize=(6, 0.5))
				plt.title('Attention Weight Color Scale')
				cmap = cm.Reds
				norm = plt.Normalize(min_val, max_val)
				cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), 
								 cax=ax, orientation='horizontal')
				cb.set_label('Attention Weight')
				plt.show()
		
		token_buttons = []
		for idx, token in enumerate(tokens):
			button = widgets.Button(
				description=tokenizer.decode([token]),
				layout=widgets.Layout(width='auto', margin='2px 2px 2px 2px')
			)
			button.token_idx = idx
			button.on_click(lambda btn: update_display(btn.token_idx, tokens))
			token_buttons.append(button)
		
		display(HTML("<div style='margin:8px 0;'><b>Click on a token below to visualize its attention (from that token back to earlier tokens):</b></div>"))
		tokens_box = widgets.HBox(token_buttons, layout=widgets.Layout(flex_flow='row wrap', width='100%'))
		display(tokens_box)
		
		output_area = widgets.Output()
		display(output_area)
		
	except ImportError:
		print("Interactive visualizer requires ipywidgets. Install with: pip install ipywidgets")

def load_llm_dataloader_mean(base_model_repo:str,lora_model_repo:str,wandb_run_name,DEVICE:str):
	api = wandb.Api()
	artifact = api.artifact(f"dmitry2-uiuc/sleeper-model-diffing/dataloader-means_run-{wandb_run_name}:latest")
	artifact_dir = Path(artifact.download(root="../../.wandb_artifacts"))
	dataloader_mean_SMPD = torch.load(artifact_dir / "dataloader_means.pt", map_location=DEVICE)

	
	llm = build_llm_lora(
		base_model_repo=base_model_repo,
		lora_model_repo=lora_model_repo,
		cache_dir=None,
		device=DEVICE,
		dtype=None
	)

	return llm,dataloader_mean_SMPD

def get_dataset(dataset_path:str,split:str='train'):
	

	dataset = load_dataset(dataset_path, split='train')
	dataset = dataset.filter(lambda x: x['is_training'] == True)
	
	return dataset

def load_xc(wandb_run_name:str,DEVICE:str):
	

	# load crosscoder decoder features
	crosscoder = load_crosscoder_from_wandb(
		"dmitry2-uiuc",
		"sleeper-model-diffing",
		wandb_run_name,
		"../../.wandb_artifacts",
		DEVICE)
	
	return crosscoder



def get_sentence_averages(layer,head,input_text,attn_class:AttentionAnalyzer):
	text_length=128
	hidden_dim=attn_class.crosscoder.hidden_dim
	data_dep_int_matrix=np.zeros((hidden_dim,hidden_dim))
	data_dep_int_matrix_abs=np.zeros((hidden_dim,hidden_dim))
	data_dep_localization_matrix=np.zeros((hidden_dim,hidden_dim))
	count=0
	for key_index in tqdm(range(text_length), disable=True):
		for query_index in range(key_index,text_length):
				feature_analysis=attn_class.analyze_feature_attention_interactions(layer,head,input_text,query_index,key_index)
				int_matrix=feature_analysis["interaction_matrix"]
				query_active_features=feature_analysis["query_active_features"]
				key_active_features=feature_analysis["key_active_features"]
				data_independent=feature_analysis["interaction_matrix_unscaled"]
				
				resized_data_dependent_int=np.zeros((hidden_dim,hidden_dim))
				resized_data_dependent_int[query_active_features[:,None],key_active_features[None,:]]=int_matrix

				resized_data_dependent_localization=np.zeros((hidden_dim,hidden_dim))
				resized_data_dependent_localization[query_active_features[:,None],key_active_features[None,:]]=np.abs(int_matrix)*(query_index-key_index)

				
				
				
				data_dep_int_matrix=data_dep_int_matrix+resized_data_dependent_int
				data_dep_int_matrix_abs=data_dep_int_matrix_abs+np.abs(resized_data_dependent_int)
				data_dep_localization_matrix=data_dep_localization_matrix+resized_data_dependent_localization

				count+=1
	
	data_dep_int_matrix/=count
	data_dep_localization_matrix=data_dep_localization_matrix/np.clip(data_dep_int_matrix_abs,a_min=1,a_max=None)
	data_dep_int_matrix_abs/=count

	return data_dep_int_matrix,data_dep_int_matrix_abs,data_dep_localization_matrix

def top_k_indices(a: np.ndarray, k: int = 10):
    """
    Return the indices (as tuples) of the k largest elements in `a`,
    in descending-value order.
    """
    # 1. Flatten -> argpartition gives positions of k largest (unsorted)
    flat_idx = np.argpartition(a.ravel(), -k)[-k:]

    # 2. Sort those k positions by actual value, descending
    flat_idx = flat_idx[np.argsort(a.ravel()[flat_idx])[::-1]]

    # 3. Convert flat indices back to n-dimensional tuples
    return np.column_stack(np.unravel_index(flat_idx, a.shape))


def get_explanations_dict(explanations_path:str):
	#csv_path = Path("/root/crosscoders-feature-interactions/sleepers/sleepers/autointerp/autointerp_data/explanations.csv")
	df = pd.read_csv(explanations_path)

	explain_dict = dict(zip(df["feature_id"], df["explanation"]))

	print(f'no. of given explanations: {len(list(explain_dict.keys()))}')
	return explain_dict

def get_averaged_tensors(attn_class, dataset, stories: int, layers: int = 4, heads: int = 16, save: bool = True) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
	"""
	Compute averaged interaction tensors across multiple stories.
	
	Args:
		attn_class: AttentionAnalyzer instance
		dataset: Dataset to analyze
		stories: Number of stories to process
		layers: Number of layers to analyze
		heads: Number of heads to analyze
		save: Whether to save the results to disk
		
	Returns:
		Dictionary mapping (layer, head) to averaged matrices
	"""
	# Input validation
	if stories <= 0:
		raise ValueError("stories must be positive")
	if layers <= 0:
		raise ValueError("layers must be positive")
	if heads <= 0:
		raise ValueError("heads must be positive")
	
	data_dict = {}
	num_heads = attn_class.llm.blocks[0].attn.W_Q.shape[0]
	hidden_dim = attn_class.crosscoder.hidden_dim
	timestamp = datetime.now().strftime("%m%d_%H%M%S")
	for layer in range(layers):
		for head in tqdm(range(heads), desc=f"Processing layer {layer}"):
			# Pre-allocate averaged matrices
			averaged_int_matrix = np.zeros((hidden_dim, hidden_dim))
			averaged_int_matrix_abs = np.zeros((hidden_dim, hidden_dim))
			averaged_int_weighted_localization = np.zeros((hidden_dim, hidden_dim))
			
			for story_idx in range(stories):
				try:
					story_matrices = get_sentence_averages(
						layer, head, dataset[story_idx]["text"], attn_class
					)
					
					avg_int_matrix, avg_int_matrix_abs, avg_int_weighted_localization = story_matrices
					
					averaged_int_matrix += avg_int_matrix
					averaged_int_matrix_abs += avg_int_matrix_abs
					averaged_int_weighted_localization += avg_int_weighted_localization
				except Exception as e:
					attn_class.logger.warning(f"Error processing story {story_idx}: {e}")
					continue
			
			# Normalize by number of stories
			averaged_int_matrix /= stories
			averaged_int_matrix_abs /= stories
			averaged_int_weighted_localization /= stories
			
			matrix_dict = {
				"avg_int_matrix": averaged_int_matrix,
				"avg_int_matrix_abs": averaged_int_matrix_abs,
				"avg_localized_matrix": averaged_int_weighted_localization
			}
			
			data_dict[(layer, head)] = matrix_dict

			# Save results if requested3
			if save:
				save_dir = Path("/root/crosscoders-feature-interactions/large_files")
				save_dir.mkdir(parents=True, exist_ok=True)
				
				save_path = save_dir / f"averaged_tensors_L{layers}_H{heads}_S{stories}_{timestamp}.pt"
				torch.save(data_dict, save_path)
				attn_class.logger.info(f"Saved averaged tensors to {save_path}")
	
	return data_dict
	

def main():
	torch.set_grad_enabled(False)
	cfg_path="attention_analysis.yaml"
	cfg=load_config(cfg_path)
	logger=setup_logger()
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
	logger.info(f"device: {DEVICE}")
	hookpoints=cfg["hookpoints"]  
	wandb_run_name=cfg["wandb_run_name"]


	base_model_repo=cfg["llm"]["base_model"]
	lora_model_repo=cfg["llm"]["lora_model"]
	llm,mean_dataloader_SMPD=load_llm_dataloader_mean(base_model_repo,lora_model_repo,wandb_run_name,DEVICE)
	logger.info(f"mean_dataloader_SMPD shape: {mean_dataloader_SMPD.shape}")


	dataset_path=cfg["dataset"]["name"]
	dataset_split=cfg["dataset"]["split"]
	dataset=get_dataset(dataset_path,dataset_split)


	xc=load_xc(wandb_run_name,DEVICE)
	
	hidden_dim=xc.hidden_dim

	attn_class=AttentionAnalyzer(xc,llm,llm.tokenizer,mean_dataloader_SMPD,hookpoints,DEVICE,logger,cfg_path)

	test=get_averaged_tensors(attn_class,dataset,stories=10,layers=4,heads=16,save=True)
	load_test=torch.load(f'/root/crosscoders-feature-interactions/large_files/averaged_tensors_L1_H1_S2.pt',weights_only=False)

	
	exit('testing averaging')
	
	
	#print(next(iter(load_test)).keys())
	#test=get_averaged_tensors(attn_class,dataset,2,layers=1,heads=1)

	

	feature_analysis=attn_class.analyze_feature_attention_interactions(layer=1,head=1,input_text=dataset[0]["text"],query_position=82,key_position=81)

	int_matrix=feature_analysis["interaction_matrix"]
	query_active_features=feature_analysis["query_active_features"]
	key_active_features=feature_analysis["key_active_features"]
	data_independent=feature_analysis["interaction_matrix_unscaled"]
	logger.info(f"interaction matrix shape {int_matrix.shape}")
	logger.info(f"query active features shape {query_active_features.shape}")
	logger.info(f"interaction matrix unscaled: {data_independent.shape}")

	
	reset_interaction_matrix=np.zeros((hidden_dim,hidden_dim))
	reset_interaction_matrix[query_active_features[:, None], key_active_features[None, :] ]=int_matrix
	logger.info(f'reset interaction matrix shape: {reset_interaction_matrix.shape}')

	




	
	#avg_int_matrix,avg_int_matrix_abs,avg_int_weighted_localization=get_sentence_averages(layer=1,head=1,input_text=dataset[0]["text"],attn_class=attn_class)
	
	#test_features=attn_class.analyze_feature_attention_interactions(layer=1,head=1,input_text=dataset[0]["text"],query_position=82,key_position=81)
	
	# data_dict=torch.load('/root/crosscoders-feature-interactions/large_files/test_ints.pt',weights_only=False)

	data_ind_matrix=attn_class.data_independent_attention(layer=1,head=1)

	print(f'data ind shape: {data_ind_matrix.shape}')
	avg_int_matrix=data_ind_matrix

	#avg_int_matrix=data_ind_matrix
	# avg_int_matrix=data_dict["avg_int_matrix"]
	# avg_int_matrix_abs=data_dict["avg_int_matrix_abs"]
	avg_int_matrix_abs_topk=top_k_indices(-np.abs(data_ind_matrix),k=10)
	
	#print(f"top 10 entries in avg matrix: {avg_int_matrix_abs_topk}")


	
	#test=attn_class.feature_interactions_whole_text(249,686,dataset[0]["text"],1,1)

	#test_int_matrix=test["interaction_matrix"]

	#largest_vals=np.sort(np.abs(test_int_matrix).flatten())[::-1]

	#print(f'largest_vals 10 {largest_vals[:10]}')


	explanations_path='/root/crosscoders-feature-interactions/sleepers/sleepers/autointerp/autointerp_data/explanations_h2mwu2g7_nohate.csv'
	explain_dict=get_explanations_dict(explanations_path)

	
	#Top two features
	for i,k in enumerate(avg_int_matrix_abs_topk):
		key_explain=None
		query_explain=None
		if k[0] in explain_dict:
			key_explain=explain_dict[k[0]]
		if k[1] in explain_dict:
			query_explain=explain_dict[k[1]]
		
		if k[0]==k[1]:
			print(print(f"Rank {i+1}: (Self-interaction), index: {k[0]} Int strength: {avg_int_matrix[k[0],k[1]]}, explanation key: {key_explain}"))
		else:
			print(f"Rank {i+1}: (distinct), indices: {k[0],k[1]} Int strength: {avg_int_matrix[k[0],k[1]]}, explanation key: {key_explain}, interaction query: {query_explain}")


	exit()

	logger.info(f"interaction matrix shape: {test_int_matrix.shape}")

	plt.imshow(test_int_matrix)


	# 3. Write directly to PDF
	out_dir = Path("/root/crosscoders-feature-interactions/large_files/graphs")           # pick any folder you wish
	out_dir.mkdir(parents=True, exist_ok=True)

	pdf_path = out_dir / "test.pdf"
	

	plt.savefig(pdf_path, format="pdf", bbox_inches="tight")



	exit()
	
	# 1. Pick where to save
	save_dir = Path("/root/crosscoders-feature-interactions/large_files")

	# 2. Ensure the directory exists (creates parents if needed)
	#os.makedirs(save_dir, exist_ok=True)
	save_dir.mkdir(parents=True, exist_ok=True)

	# 3. Package the tensors
	data_dict = {
		"avg_int_matrix": avg_int_matrix,
		"avg_int_matrix_abs": avg_int_matrix_abs,
		"avg_int_weighted_localization": avg_int_weighted_localization,
		"input_text":dataset[0]["text"]
		}

	# 4. Save them
	file_path = save_dir / "test_ints.pt"
	
	print(f'saved to: {file_path}')
	torch.save(data_dict, file_path)

	
	

def quick_plot(data_dict):
	avg_int_matrix=data_dict["avg_int_matrix"]
	avg_int_matrix_abs=data_dict["avg_int_matrix_abs"]
	avg_int_weighted_localization=data_dict["avg_int_weighted_localization"]

	vals = avg_int_matrix[avg_int_matrix != 0].flatten()     # or torch.flatten(...).cpu().numpy()

	# 2. Create the histogram
	fig, ax = plt.subplots()        # adjust size as you like
	ax.hist(vals, bins="auto", edgecolor="black")  # automatic bin count
	ax.set_xlabel("Value")
	ax.set_ylabel("Frequency")
	ax.set_title("Histogram of avg_int_matrix values")

	# 3. Write directly to PDF
	out_dir = Path("/root/crosscoders-feature-interactions/large_files/graphs")           # pick any folder you wish
	out_dir.mkdir(parents=True, exist_ok=True)

	pdf_path = out_dir / "test.pdf"
	fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

	print("✓  PDF written to:", pdf_path.resolve())
	



if __name__== "__main__":
	print('the main character')
	main()
	#data_dict=torch.load("/root/crosscoders-feature-interactions/large_files/test_ints.pt",weights_only=False)

	