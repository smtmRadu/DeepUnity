import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import json
import pickle

class HFModelDebugger:
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.operation_log = []
        self.tensor_cache = OrderedDict()
        self.hooks = []
        self.operation_counter = 0
        
    def clear(self):
        """Remove all hooks and clear logs"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.operation_log.clear()
        self.tensor_cache.clear()
        self.operation_counter = 0
    
    def tensor_stats(self, tensor, name=""):
        """Get comprehensive tensor statistics - FIXED for integer tensors"""
        if not isinstance(tensor, torch.Tensor):
            return {"type": str(type(tensor)), "value": str(tensor)[:100]}
        
        stats = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "numel": tensor.numel(),
        }
        
        if tensor.numel() > 0:
            flat_tensor = tensor.flatten()
            stats.update({
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "first_10": flat_tensor[:10].tolist(),
                "last_10": flat_tensor[-10:].tolist(),
                "zero_count": (tensor == 0).sum().item(),
            })
            
            # Handle floating point vs integer tensors differently
            if tensor.dtype.is_floating_point:
                stats.update({
                    "mean": tensor.mean().item(),
                    "std": tensor.std().item(),
                    "norm": tensor.norm().item(),
                    "nan_count": torch.isnan(tensor).sum().item(),
                    "inf_count": torch.isinf(tensor).sum().item()
                })
            elif tensor.dtype.is_complex:
                stats.update({
                    "mean": tensor.mean().item(),
                    "std": tensor.std().item(),
                    "norm": tensor.norm().item(),
                    "nan_count": torch.isnan(tensor).sum().item(),
                    "inf_count": torch.isinf(tensor).sum().item()
                })
            else:
                # For integer tensors, convert to float for statistics
                float_tensor = tensor.float()
                stats.update({
                    "mean": float_tensor.mean().item(),
                    "std": float_tensor.std().item() if tensor.numel() > 1 else 0.0,
                    "norm": float_tensor.norm().item(),
                    "nan_count": 0,  # Integer tensors can't have NaN
                    "inf_count": 0   # Integer tensors can't have Inf
                })
        
        return stats

    def make_detailed_hook(self, module_name, module_type):
        def hook_fn(module, input_tensors, output):
            # Only log if module matches target patterns
            name_lower = module_name.lower()
            type_lower = module_type.lower()
            target_patterns = ['q_norm', 'k_norm', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'mlp', 'attention', 'feed_forward', 'embed_tokens', 'input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm', 'post_feedforward_layernorm']
            if not any(p in name_lower or p in type_lower for p in target_patterns):
                return  # Skip logging for non-target modules

            op_id = f"{self.operation_counter:03d}_{module_name}"
            self.operation_counter += 1
            
            # Process inputs
            input_stats = []
            if isinstance(input_tensors, tuple):
                for i, inp in enumerate(input_tensors):
                    if isinstance(inp, torch.Tensor):
                        stats = self.tensor_stats(inp, f"{module_name}_input_{i}")
                        input_stats.append(stats)
                        self.tensor_cache[f"{op_id}_input_{i}"] = inp.detach().clone()
            elif isinstance(input_tensors, torch.Tensor):
                stats = self.tensor_stats(input_tensors, f"{module_name}_input")
                input_stats.append(stats)
                self.tensor_cache[f"{op_id}_input"] = input_tensors.detach().clone()
            
            # Process outputs
            output_stats = []
            if isinstance(output, tuple):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        stats = self.tensor_stats(out, f"{module_name}_output_{i}")
                        output_stats.append(stats)
                        self.tensor_cache[f"{op_id}_output_{i}"] = out.detach().clone()
            elif isinstance(output, torch.Tensor):
                stats = self.tensor_stats(output, f"{module_name}_output")
                output_stats.append(stats)
                self.tensor_cache[f"{op_id}_output"] = output.detach().clone()
            
            # Log the operation
            operation = {
                "id": op_id,
                "order": self.operation_counter,
                "module_name": module_name,
                "module_type": module_type,
                "module_class": module.__class__.__name__,
                "inputs": input_stats,
                "outputs": output_stats,
                "parameters": {}
            }
            
            # Add parameter info
            if hasattr(module, 'weight') and module.weight is not None:
                operation["parameters"]["weight"] = self.tensor_stats(module.weight, f"{module_name}_weight")
            if hasattr(module, 'bias') and module.bias is not None:
                operation["parameters"]["bias"] = self.tensor_stats(module.bias, f"{module_name}_bias")
            
            self.operation_log.append(operation)
            
            # >>> PRINT FULL DETAILED STATS (FIRST 10, LAST 10) <<<
            print(f"\n{'='*80}")
            print(f"{op_id}: {module_type} ({module_name})")
            print(f"{'='*80}")
            
            for i, inp_stat in enumerate(input_stats):
                if 'shape' in inp_stat:
                    print(f"\n  → INPUT[{i}]: {inp_stat['name']}")
                    print(f"     Shape: {inp_stat['shape']}")
                    print(f"     Dtype: {inp_stat['dtype']}")
                    if 'mean' in inp_stat:
                        print(f"     Mean: {inp_stat['mean']:.6f}, Std: {inp_stat['std']:.6f}")
                    else:
                        print(f"     Min: {inp_stat['min']}, Max: {inp_stat['max']}")
                    print(f"     First 10: {inp_stat['first_10']}")
                    print(f"     Last 10:  {inp_stat['last_10']}")
                    print(f"     Zeros: {inp_stat['zero_count']}, Total: {inp_stat['numel']}")
            
            for i, out_stat in enumerate(output_stats):
                if 'shape' in out_stat:
                    print(f"\n  → OUTPUT[{i}]: {out_stat['name']}")
                    print(f"     Shape: {out_stat['shape']}")
                    print(f"     Dtype: {out_stat['dtype']}")
                    if 'mean' in out_stat:
                        print(f"     Mean: {out_stat['mean']:.6f}, Std: {out_stat['std']:.6f}")
                    else:
                        print(f"     Min: {out_stat['min']}, Max: {out_stat['max']}")
                    print(f"     First 10: {out_stat['first_10']}")
                    print(f"     Last 10:  {out_stat['last_10']}")
                    print(f"     Zeros: {out_stat['zero_count']}, Total: {out_stat['numel']}")
            
            # Print parameters if any
            for param_name, param_info in operation["parameters"].items():
                if 'shape' in param_info:
                    print(f"\n  → PARAMETER: {param_name}")
                    print(f"     Shape: {param_info['shape']}")
                    print(f"     Mean: {param_info['mean']:.6f}")
                    print(f"     First 10: {param_info['first_10']}")
                    print(f"     Last 10:  {param_info['last_10']}")

        return hook_fn

    def register_all_hooks(self, target_layers=None):
        """Register hooks on ALL modules — filter by name inside hook"""
        print("Registering hooks on ALL modules... (filtering by name in hook)")

        for name, module in self.model.named_modules():
            hook = module.register_forward_hook(
                self.make_detailed_hook(name, module.__class__.__name__)
            )
            self.hooks.append(hook)
        
        print(f"Registered {len(self.hooks)} hooks (filtering applied in hook function)")

    def debug_forward_pass(self, input_text=None, input_ids=None, max_length=50):
        """Run a forward pass and collect all operation details"""
        self.clear()
        self.register_all_hooks()
        
        try:
            # Prepare input
            if input_text and self.tokenizer:
                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)
                input_ids = inputs['input_ids']
                print(f"Input text: '{input_text}'")
                print(f"Tokenized: {self.tokenizer.convert_ids_to_tokens(input_ids[0])}")
            elif input_ids is None:
                # Default input for testing
                input_ids = torch.tensor([[1, 2, 3, 4, 5]])
            
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Input IDs: {input_ids}")
            
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                if hasattr(self.model, 'generate'):
                    outputs = self.model(input_ids, output_hidden_states=True, output_attentions=True)
                else:
                    outputs = self.model(input_ids)
            
            print(f"\nForward pass completed. Tracked {len(self.operation_log)} operations.")
            
            return outputs, input_ids
            
        finally:
            pass  # Keep hooks for multiple passes if needed

    def compare_tensor_with_yours(self, your_tensor, operation_id, tensor_type="output", index=0):
        """Compare your implementation's tensor with the HF model's tensor"""
        cache_key = f"{operation_id}_{tensor_type}"
        if index > 0:
            cache_key += f"_{index}"
        
        if cache_key not in self.tensor_cache:
            available_keys = [k for k in self.tensor_cache.keys() if operation_id in k]
            print(f"❌ Key '{cache_key}' not found.")
            print(f"Available keys for {operation_id}: {available_keys}")
            return False
        
        hf_tensor = self.tensor_cache[cache_key]
        
        # Shape comparison
        if your_tensor.shape != hf_tensor.shape:
            print(f"❌ Shape mismatch: yours={your_tensor.shape}, HF={hf_tensor.shape}")
            return False
        
        # Value comparison
        max_diff = torch.max(torch.abs(your_tensor - hf_tensor)).item()
        mean_diff = torch.mean(torch.abs(your_tensor - hf_tensor)).item()
        rel_error = mean_diff / (torch.mean(torch.abs(hf_tensor)).item() + 1e-8)
        
        print(f"\n=== Comparison for {cache_key} ===")
        print(f"Shape: {your_tensor.shape} ✓")
        print(f"Max absolute difference: {max_diff:.8f}")
        print(f"Mean absolute difference: {mean_diff:.8f}")
        print(f"Relative error: {rel_error:.8f}")
        
        # Detailed stats
        print(f"\nYOUR tensor:")
        print(f"  Mean: {your_tensor.mean():.8f}, Std: {your_tensor.std():.8f}")
        print(f"  Min: {your_tensor.min():.8f}, Max: {your_tensor.max():.8f}")
        print(f"HF tensor:")
        print(f"  Mean: {hf_tensor.mean():.8f}, Std: {hf_tensor.std():.8f}")
        print(f"  Min: {hf_tensor.min():.8f}, Max: {hf_tensor.max():.8f}")
        
        # First few values comparison
        print(f"\nFirst 5 values comparison:")
        your_flat = your_tensor.flatten()[:5]
        hf_flat = hf_tensor.flatten()[:5]
        for i in range(min(5, len(your_flat))):
            print(f"  [{i}] Yours: {your_flat[i]:.8f}, HF: {hf_flat[i]:.8f}, Diff: {abs(your_flat[i] - hf_flat[i]):.8f}")
        
        # Tolerance check
        is_close = torch.allclose(your_tensor, hf_tensor, rtol=1e-5, atol=1e-6)
        print(f"\n{'✅' if is_close else '❌'} Close within tolerance (rtol=1e-5, atol=1e-6): {is_close}")
        
        if not is_close and max_diff > 1e-3:
            print("⚠️  Large differences detected! Check your implementation.")
        elif not is_close and max_diff > 1e-5:
            print("⚠️  Small differences detected. Might be numerical precision or different implementations.")
        
        return is_close

    def get_operation_by_name(self, name_pattern):
        """Find operations by name pattern"""
        matches = [op for op in self.operation_log if name_pattern.lower() in op['module_name'].lower()]
        return matches

    def print_operation_summary(self, max_ops=None):
        """Print a summary of all operations"""
        ops_to_show = self.operation_log[:max_ops] if max_ops else self.operation_log
        
        print(f"\n{'='*80}")
        print(f"OPERATION SUMMARY ({len(ops_to_show)} operations)")
        print(f"{'='*80}")
        
        for op in ops_to_show:
            print(f"\n{op['id']}: {op['module_name']} ({op['module_type']})")
            
            for i, inp in enumerate(op['inputs']):
                if 'shape' in inp:
                    dtype_str = inp.get('dtype', '')
                    if 'float' in dtype_str.lower() or 'double' in dtype_str.lower():
                        print(f"  IN[{i}]:  {inp['shape']} | μ={inp['mean']:.6f} σ={inp['std']:.6f}")
                    else:
                        print(f"  IN[{i}]:  {inp['shape']} | min={inp['min']} max={inp['max']} | {dtype_str}")
                    print(f"    First 10: {inp['first_10']}")
                    print(f"    Last 10:  {inp['last_10']}")
            
            for i, out in enumerate(op['outputs']):
                if 'shape' in out:
                    dtype_str = out.get('dtype', '')
                    if 'float' in dtype_str.lower() or 'double' in dtype_str.lower():
                        print(f"  OUT[{i}]: {out['shape']} | μ={out['mean']:.6f} σ={out['std']:.6f}")
                    else:
                        print(f"  OUT[{i}]: {out['shape']} | min={out['min']} max={out['max']} | {dtype_str}")
                    print(f"    First 10: {out['first_10']}")
                    print(f"    Last 10:  {out['last_10']}")
            
            for param_name, param_info in op['parameters'].items():
                if 'shape' in param_info:
                    print(f"  {param_name.upper()}: {param_info['shape']} | μ={param_info['mean']:.6f}")

    def export_debug_data(self, filename_prefix="debug"):
        """Export all debugging data"""
        with open(f"{filename_prefix}_operations.json", 'w') as f:
            json.dump(self.operation_log, f, indent=2, default=str)
        
        with open(f"{filename_prefix}_tensors.pkl", 'wb') as f:
            cpu_cache = {k: v.cpu().detach() for k, v in self.tensor_cache.items()}
            pickle.dump(cpu_cache, f)
        
        print(f"Debug data exported to {filename_prefix}_operations.json and {filename_prefix}_tensors.pkl")

    def load_debug_data(self, filename_prefix="debug"):
        """Load previously exported debug data"""
        with open(f"{filename_prefix}_operations.json", 'r') as f:
            self.operation_log = json.load(f)
        
        with open(f"{filename_prefix}_tensors.pkl", 'rb') as f:
            self.tensor_cache = pickle.load(f)
        
        print(f"Debug data loaded from {filename_prefix}_*")


def debug_huggingface_model(model, tokenizer=None, input_text="Hello world", save_debug_data=True):
    debugger = HFModelDebugger(model, tokenizer)
    
    try:
        outputs, input_ids = debugger.debug_forward_pass(input_text=input_text)
        debugger.print_operation_summary()
        
        if save_debug_data:
            debugger.export_debug_data("hf_model_debug")
        
        return debugger, outputs, input_ids
    
    except Exception as e:
        print(f"Error during debugging: {e}")
        raise
    finally:
        debugger.clear()


def compare_implementations_step_by_step(debugger, your_implementation_outputs):
    print("\n" + "="*80)
    print("STEP-BY-STEP COMPARISON")
    print("="*80)
    
    matches = 0
    total = 0
    
    for op_id, your_tensor in your_implementation_outputs.items():
        print(f"\n--- Checking {op_id} ---")
        is_match = debugger.compare_tensor_with_yours(your_tensor, op_id)
        
        if is_match:
            matches += 1
        total += 1
    
    print(f"\n{'='*40}")
    print(f"FINAL SCORE: {matches}/{total} operations match")
    print(f"Success rate: {matches/total*100:.1f}%")
    print(f"{'='*40}")
    
    return matches, total