import json, sys
sys.stdout.reconfigure(encoding='utf-8')

data = [json.loads(l) for l in open('results/translations.jsonl', encoding='utf-8')]

with open('results/sample_translations.txt', 'w', encoding='utf-8') as f:
    for d in data:
        f.write(f"[{d['domain'].upper()}] ID={d['id']}\n")
        f.write(f"  SRC : {d['source']}\n")
        f.write(f"  HYP : {d['hypothesis']}\n")
        f.write(f"  REF : {d['reference']}\n")
        comet = f"{d['comet_score']:.4f}" if d['comet_score'] else "N/A"
        f.write(f"  COMET: {comet}\n")
        f.write("\n")

print("Written to results/sample_translations.txt")
