
"""
LoRA Ablation Utilities

This module provides functionality to selectively disable LoRA adapters during training
for ablation studies. It supports both standard PEFT LoRA and StreamAwareLoRA modules.

The ablation works by filtering the LoRA configuration before model creation:
1. **Module-level ablation**: Remove specific modules (q_proj, k_proj, etc.) from target_modules
2. **Layer-level ablation**: Use layers_to_transform to exclude specific layers

This approach is clean and efficient as LoRA adapters are never created for ablated
modules/layers in the first place.
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def filter_lora_modules_for_ablation(
    lora_modules: List[str],
    enabled: bool = False,
    ablated_modules: List[str] = None,
    ablated_layers: List[int] = None,
    total_layers: int = None,
) -> Tuple[List[str], Optional[List[int]]]:

    if not enabled:
        return lora_modules, None
    
    if not total_layers:
        raise("Total Layers missing, must be passed in prior to ablation filtering.")
   
    if not ablated_modules and not ablated_layers:
        logger.warning("LoRA ablation enabled but no modules or layers specified")
        return lora_modules, None
    
    # Start with all modules
    filtered_modules = lora_modules.copy()
    layers_to_transform = None
    
    # Remove modules based on module patterns
    if ablated_modules:
        for ablated_module in ablated_modules:
            if ablated_module in filtered_modules:
                filtered_modules.remove(ablated_module)
                logger.info(f"Removed module '{ablated_module}' from LoRA target_modules")
            else:
                logger.warning(f"Module '{ablated_module}' not found in target_modules: {lora_modules}")
    
    # Handle layer-based ablation using layers_to_transform
    layers_to_transform = list(set(range(total_layers)) - set(ablated_layers))
    logger.info(f"Layer-based ablation: excluding layers {ablated_layers}, transforming layers {layers_to_transform}")
    
    return filtered_modules, layers_to_transform