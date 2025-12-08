"""
Compare demo checkpoints from old and new trackB code
"""
import torch

# Paths
OLD_CKPT = r"d:\Thesis\Ego4d-LiteSTA\local_extraction_old\trackB\old_runs\trackB_final_20251208_040553.pt"
NEW_CKPT = r"d:\Thesis\Ego4d-LiteSTA\local_extraction\runs\Track_B\demo_comparison\trackB_demo_20251208_040757.pt"

def get_param_stats(state_dict):
    """Get parameter statistics"""
    total_params = 0
    total_norm = 0.0
    for k, v in state_dict.items():
        norm = v.float().norm().item()
        total_params += v.numel()
        total_norm += norm
    return total_params, total_norm

def compare_checkpoints():
    print("DEMO CHECKPOINT COMPARISON")
    print("="*70)
    
    # Load checkpoints
    old_ckpt = torch.load(OLD_CKPT, map_location='cpu')
    new_ckpt = torch.load(NEW_CKPT, map_location='cpu')
    
    print("\n1. CHECKPOINT KEYS")
    print("-"*40)
    print(f"OLD: {sorted(old_ckpt.keys())}")
    print(f"NEW: {sorted(new_ckpt.keys())}")
    
    print("\n2. FUSION PARAMETERS")
    print("-"*40)
    old_f_params, old_f_norm = get_param_stats(old_ckpt['fusion'])
    new_f_params, new_f_norm = get_param_stats(new_ckpt['fusion'])
    
    print(f"OLD: {old_f_params:,} params, total norm: {old_f_norm:.4f}")
    print(f"NEW: {new_f_params:,} params, total norm: {new_f_norm:.4f}")
    
    # Compute weight differences
    print("\nFusion weight differences:")
    common_keys = set(old_ckpt['fusion'].keys()) & set(new_ckpt['fusion'].keys())
    for k in sorted(common_keys)[:6]:
        old_v = old_ckpt['fusion'][k].float()
        new_v = new_ckpt['fusion'][k].float()
        diff = (old_v - new_v).abs().mean().item()
        print(f"  {k}: diff={diff:.6f}")
    
    print("\n3. HEAD PARAMETERS")
    print("-"*40)
    old_h_params, old_h_norm = get_param_stats(old_ckpt['head'])
    new_h_params, new_h_norm = get_param_stats(new_ckpt['head'])
    
    print(f"OLD: {old_h_params:,} params, total norm: {old_h_norm:.4f}")
    print(f"NEW: {new_h_params:,} params, total norm: {new_h_norm:.4f}")
    
    # Check noun/verb head specifically
    print("\nHead weight differences:")
    for k in ['noun_head.weight', 'noun_head.bias', 'verb_head.weight', 'verb_head.bias', 'ttc_head.weight', 'ttc_head.bias']:
        if k in old_ckpt['head'] and k in new_ckpt['head']:
            old_v = old_ckpt['head'][k].float()
            new_v = new_ckpt['head'][k].float()
            diff = (old_v - new_v).abs().mean().item()
            print(f"  {k}: diff={diff:.6f}")
    
    print("\n4. PROJECTOR PARAMETERS")
    print("-"*40)
    old_p_params, old_p_norm = get_param_stats(old_ckpt['projector'])
    new_p_params, new_p_norm = get_param_stats(new_ckpt['projector'])
    
    print(f"OLD: {old_p_params:,} params, total norm: {old_p_norm:.4f}")
    print(f"NEW: {new_p_params:,} params, total norm: {new_p_norm:.4f}")
    
    for k in old_ckpt['projector']:
        if k in new_ckpt['projector']:
            old_v = old_ckpt['projector'][k].float()
            new_v = new_ckpt['projector'][k].float()
            diff = (old_v - new_v).abs().mean().item()
            print(f"  {k}: diff={diff:.6f}")
    
    print("\n5. OVERALL SUMMARY")
    print("-"*40)
    
    # Compute total weight difference
    total_diff = 0.0
    total_count = 0
    
    for module in ['fusion', 'head', 'projector']:
        for k in old_ckpt[module]:
            if k in new_ckpt[module]:
                old_v = old_ckpt[module][k].float()
                new_v = new_ckpt[module][k].float()
                diff = (old_v - new_v).abs().sum().item()
                total_diff += diff
                total_count += old_v.numel()
    
    avg_diff = total_diff / total_count if total_count > 0 else 0
    print(f"Average weight difference: {avg_diff:.8f}")
    print(f"Total absolute difference: {total_diff:.4f}")
    
    # Are they identical?
    if total_diff < 1e-5:
        print("\n>>> WEIGHTS ARE ESSENTIALLY IDENTICAL!")
    else:
        print("\n>>> WEIGHTS ARE DIFFERENT")
        print(f"    This indicates different training dynamics")

if __name__ == "__main__":
    compare_checkpoints()
