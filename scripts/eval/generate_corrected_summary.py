"""Generate corrected summary with NEI=Neutral accuracy."""
import json
from pathlib import Path

def main():
    # Load corrected data
    with open('results/seed1000_topic_accuracy_corrected.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get OVERALL if exists, otherwise calculate
    if 'OVERALL' in data:
        overall = data['OVERALL']
    else:
        # Calculate overall
        total_all = sum(s['total'] for s in data.values())
        correct_standard_all = sum(s['correct_standard'] for s in data.values())
        correct_nei_all = sum(s['correct_nei_neutral'] for s in data.values())
        total_nei_all = sum(s['total_nei_neutral'] for s in data.values())
        overall = {
            'total': total_all,
            'standard_accuracy': (correct_standard_all / total_all * 100) if total_all > 0 else 0.0,
            'nei_neutral_accuracy': (correct_nei_all / total_nei_all * 100) if total_nei_all > 0 else 0.0,
            'correct_standard': correct_standard_all,
            'correct_nei_neutral': correct_nei_all,
            'total_nei_neutral': total_nei_all,
        }
    
    # Sort topics
    topics = sorted([k for k in data.keys() if k != 'OVERALL'])
    
    # Generate markdown table
    lines = []
    lines.append("# Accuracy per Topic - seed1000 Cold Run (CORRECTED)")
    lines.append("")
    lines.append("**Dataset:** seed1000_cold.jsonl (1000 test cases)")
    lines.append("**Evaluation Date:** After Phase 2 completion")
    lines.append("**Note:** NEI=Neutral ignores claims where predicted=NOT_ENOUGH_INFO (only counts decisions)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Topic | Total | Standard Accuracy | NEI=Neutral Accuracy | Correct (Std) | Correct (NEI) |")
    lines.append("|-------|-------|-------------------|----------------------|---------------|---------------|")
    
    for topic in topics:
        t = data[topic]
        lines.append(f"| **{topic}** | {t['total']} | {t['standard_accuracy']:.2f}% | {t['nei_neutral_accuracy']:.2f}% | {t['correct_standard']}/{t['total']} | {t['correct_nei_neutral']}/{t['total_nei_neutral']} |")
    
    lines.append(f"| **OVERALL** | **{overall['total']}** | **{overall['standard_accuracy']:.2f}%** | **{overall['nei_neutral_accuracy']:.2f}%** | **{overall['correct_standard']}/{overall['total']}** | **{overall['correct_nei_neutral']}/{overall['total_nei_neutral']}** |")
    
    # Save
    output_path = Path('results/seed1000_topic_accuracy_summary_corrected.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Summary saved to: {output_path}")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
