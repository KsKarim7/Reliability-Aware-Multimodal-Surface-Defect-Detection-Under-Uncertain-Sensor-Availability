
state_anomaly = ["damaged {}",
                 "flawed {}",
                 "abnormal {}",
                 "imperfect {}",
                 "blemished {}",
                 "{} with flaw",
                 "{} with defect",
                 "{} with damage"]

abnormal_state0 = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

class_state_abnormal = {
    'bagel': ['{} with defect', '{} with contamination', '{} with crack', '{} with hole'],
    'cable_gland': ['{} with a bent shape', '{} with cut', '{} with hole', '{} with thread residue'],
    'carrot': ['{} with defect', '{} with contamination', '{} with crack', '{} with cut', '{} with hole'],
    'cookie': ['{} with defect', '{} with contamination', '{} with crack', '{} with hole'],
    'dowel': ['{} with a bent shape', '{} with defect', '{} with contamination', '{} with cut'],
    'foam': ['{} with color spot', '{} with defect', '{} with contamination', '{} with cut'],
    'peach': ['{} with defect', '{} with contamination', '{} with cut', '{} with hole'],
    'potato': ['{} with defect', '{} with contamination', '{} with cut', '{} with hole'],
    'rope': ['{} with contamination', '{} with cut', '{} with open part'],
    'tire': ['{} with defect', '{} with contamination', '{} with cut', '{} with hole']
}

# Innovation 3: Multi-granularity text templates per layer depth
# coarse  → early layers  (layers 0-1): broad detection signal
# mid     → middle layers (layers 2-3): defect category signal
# fine    → deep layers   (layers 4-5): specific surface signal
class_state_granular = {
    'bagel': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with crack', '{} with hole', '{} with contamination'],
        'fine':   ['{} with surface crack', '{} with surface hole', '{} with surface contamination at edge']
    },
    'cable_gland': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with cut', '{} with hole', '{} with bent shape'],
        'fine':   ['{} with surface cut', '{} with thread residue', '{} with bent shape at edge']
    },
    'carrot': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with crack', '{} with cut', '{} with hole'],
        'fine':   ['{} with surface crack', '{} with surface cut at edge', '{} with surface contamination']
    },
    'cookie': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with crack', '{} with hole', '{} with contamination'],
        'fine':   ['{} with surface crack', '{} with surface hole at edge', '{} with surface contamination']
    },
    'dowel': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with cut', '{} with contamination', '{} with bent shape'],
        'fine':   ['{} with surface cut', '{} with surface contamination at edge', '{} with bent shape']
    },
    'foam': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with cut', '{} with color spot', '{} with contamination'],
        'fine':   ['{} with surface cut at edge', '{} with color spot on surface', '{} with surface contamination']
    },
    'peach': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with cut', '{} with hole', '{} with contamination'],
        'fine':   ['{} with surface cut', '{} with surface hole at edge', '{} with surface contamination']
    },
    'potato': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with cut', '{} with hole', '{} with contamination'],
        'fine':   ['{} with surface cut', '{} with surface hole', '{} with surface contamination at edge']
    },
    'rope': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with cut', '{} with open part', '{} with contamination'],
        'fine':   ['{} with surface cut', '{} with open part at edge', '{} with surface contamination']
    },
    'tire': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with cut', '{} with hole', '{} with contamination'],
        'fine':   ['{} with surface cut', '{} with surface hole at edge', '{} with surface contamination']
    }
}

# Eyescandies class prompts
class_state_abnormal.update({
    'CandyCane': ['{} with defect', '{} with bump', '{} with dent', '{} with color spot'],
    'ChocolateCookie': ['{} with defect', '{} with bump', '{} with dent', '{} with crack'],
    'ChocolatePraline': ['{} with defect', '{} with bump', '{} with dent', '{} with color spot'],
    'Confetto': ['{} with defect', '{} with bump', '{} with dent', '{} with color spot'],
    'GummyBear': ['{} with defect', '{} with bump', '{} with dent', '{} with color spot'],
    'HazelnutTruffle': ['{} with defect', '{} with bump', '{} with dent', '{} with crack'],
    'LicoriceSandwich': ['{} with defect', '{} with bump', '{} with dent', '{} with color spot'],
    'Lollipop': ['{} with defect', '{} with bump', '{} with dent', '{} with color spot'],
    'Marshmallow': ['{} with defect', '{} with bump', '{} with dent', '{} with color spot'],
    'PeppermintCandy': ['{} with defect', '{} with bump', '{} with dent', '{} with color spot'],
})

class_state_granular.update({
    'CandyCane': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with bump', '{} with dent', '{} with color spot'],
        'fine':   ['{} with surface bump', '{} with surface dent', '{} with color spot on surface']
    },
    'ChocolateCookie': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with bump', '{} with dent', '{} with crack'],
        'fine':   ['{} with surface bump', '{} with surface dent', '{} with surface crack']
    },
    'ChocolatePraline': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with bump', '{} with dent', '{} with color spot'],
        'fine':   ['{} with surface bump', '{} with surface dent', '{} with color spot on surface']
    },
    'Confetto': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with bump', '{} with dent', '{} with color spot'],
        'fine':   ['{} with surface bump', '{} with surface dent', '{} with color spot on surface']
    },
    'GummyBear': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with bump', '{} with dent', '{} with color spot'],
        'fine':   ['{} with surface bump', '{} with surface dent', '{} with color spot on surface']
    },
    'HazelnutTruffle': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with bump', '{} with dent', '{} with crack'],
        'fine':   ['{} with surface bump', '{} with surface dent', '{} with surface crack']
    },
    'LicoriceSandwich': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with bump', '{} with dent', '{} with color spot'],
        'fine':   ['{} with surface bump', '{} with surface dent', '{} with color spot on surface']
    },
    'Lollipop': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with bump', '{} with dent', '{} with color spot'],
        'fine':   ['{} with surface bump', '{} with surface dent', '{} with color spot on surface']
    },
    'Marshmallow': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with bump', '{} with dent', '{} with color spot'],
        'fine':   ['{} with surface bump', '{} with surface dent', '{} with color spot on surface']
    },
    'PeppermintCandy': {
        'coarse': ['{} with defect'],
        'mid':    ['{} with bump', '{} with dent', '{} with color spot'],
        'fine':   ['{} with surface bump', '{} with surface dent', '{} with color spot on surface']
    },
})
