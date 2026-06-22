import shutil
import os

src = '/home/p3766/MISDD-MM/MISDD_MM/model.py'
backup = '/home/p3766/MISDD-MM/MISDD_MM/model_full.py'
shutil.copy(src, backup)

def read_model():
    with open(backup, 'r') as f:
        return f.readlines()

def comment_single(lines, partial):
    result = []
    for line in lines:
        stripped = line.lstrip()
        if partial in line and not stripped.startswith('#'):
            indent = line[:len(line)-len(stripped)]
            result.append(indent + '# ' + stripped)
        else:
            result.append(line)
    return result

def replace_sentinel_with_fallback(lines):
    """Replace the entire sensor sentinel block with simple fallback assignments"""
    result = []
    skip = False
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect start of sentinel block
        if 'w_rgb, w_dep = self.sensor_sentinel.get_quality_weights(' in line:
            skip = True
            indent = line[:len(line)-len(line.lstrip())]
            # Replace with fallback assignments
            result.append(indent + 'initial_prompt_image = self.image_prompt_complete\n')
            result.append(indent + 'initial_prompt_depth = self.depth_prompt_complete\n')
            result.append(indent + 'common_prompt = self.common_prompt_complete\n')
        # Detect end of sentinel block
        if skip and 'w_dep * self.common_prompt_depth)' in line:
            skip = False
            i += 1
            continue
        if not skip:
            result.append(line)
        i += 1
    return result

def comment_block(lines, start_partial, end_partial):
    result = []
    in_block = False
    for line in lines:
        stripped = line.lstrip()
        if start_partial in line and not stripped.startswith('#'):
            in_block = True
        if in_block:
            indent = line[:len(line)-len(stripped)]
            result.append(indent + '# ' + stripped)
            if end_partial in line:
                in_block = False
        else:
            result.append(line)
    return result

def fix_empty_if_blocks(lines):
    result = []
    for i, line in enumerate(lines):
        result.append(line)
        stripped = line.rstrip()
        if stripped.endswith(':') and ('if ' in line or 'else:' in line or 'elif ' in line):
            current_indent = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines) and (lines[j].strip() == '' or lines[j].strip().startswith('#')):
                j += 1
            if j < len(lines):
                next_indent = len(lines[j]) - len(lines[j].lstrip())
                if next_indent <= current_indent:
                    result.append(' ' * (current_indent + 4) + 'pass\n')
    return result

def make_variant(disable_sentinel=False, disable_dynamic=False,
                 disable_correlated=False, disable_granular=False):
    lines = read_model()

    if disable_sentinel:
        lines = replace_sentinel_with_fallback(lines)

    if disable_dynamic:
        lines = comment_block(lines,
            'dyn_image = self.dynamic_image_gen(',
            'base_image = base_image + dyn_image')
        lines = comment_block(lines,
            'dyn_depth = self.dynamic_depth_gen(',
            'base_depth = base_depth + dyn_depth')

    if disable_correlated:
        lines = comment_single(lines, 'corr_image = self.correlated_prompt_image')
        lines = comment_single(lines, 'corr_depth = self.correlated_prompt_depth')
        lines = comment_single(lines, 'all_prompts_image[index].append(cross_image + corr_image)')
        lines = comment_single(lines, 'all_prompts_depth[index].append(cross_depth + corr_depth)')

    if disable_granular:
        lines = comment_single(lines, 'self.granular_text_guidance = GranularTextGuidance')

    lines = fix_empty_if_blocks(lines)
    return lines

configs = {
    'innov1_only': dict(disable_sentinel=True,  disable_dynamic=True,  disable_correlated=False, disable_granular=True),
    'innov2_only': dict(disable_sentinel=True,  disable_dynamic=False, disable_correlated=True,  disable_granular=True),
    'innov3_only': dict(disable_sentinel=True,  disable_dynamic=True,  disable_correlated=True,  disable_granular=False),
    'innov4_only': dict(disable_sentinel=False, disable_dynamic=True,  disable_correlated=True,  disable_granular=True),
    'innov1_3':    dict(disable_sentinel=True,  disable_dynamic=True,  disable_correlated=False, disable_granular=False),
    'innov1_4':    dict(disable_sentinel=False, disable_dynamic=True,  disable_correlated=False, disable_granular=True),
}

os.makedirs('/home/p3766/MISDD-MM/ablation_models', exist_ok=True)
for name, cfg in configs.items():
    lines = make_variant(**cfg)
    out = f'/home/p3766/MISDD-MM/ablation_models/model_{name}.py'
    with open(out, 'w') as f:
        f.writelines(lines)
    print(f"Created: {out}")

shutil.copy(backup, src)
print("Original model.py restored")
print("Done")
