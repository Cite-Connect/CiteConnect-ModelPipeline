#!/bin/bash
# Quick script to check hyperparameter tuning progress

echo "Checking hyperparameter tuning progress..."
echo ""

# Check if results file exists and has content
if [ -f "bias_config/best_hyperparameters_cold_start.json" ]; then
    SIZE=$(stat -f%z "bias_config/best_hyperparameters_cold_start.json" 2>/dev/null || stat -c%s "bias_config/best_hyperparameters_cold_start.json" 2>/dev/null || echo "0")
    if [ "$SIZE" -gt 100 ]; then
        echo "✅ Script completed! Results file has content."
        echo ""
        python3 -c "
import json
try:
    with open('bias_config/best_hyperparameters_cold_start.json', 'r') as f:
        d = json.load(f)
    print('📊 Results Summary:')
    print(f'  Configs tested: {d.get(\"num_configurations\", 0)}')
    print(f'  Users evaluated: {d.get(\"user_sample_size\", 0)}')
    print('')
    print('🏆 Best Configuration:')
    bc = d.get('best_config', {})
    print(f'  Name: {bc.get(\"name\", \"N/A\")}')
    print(f'  Weights: {bc.get(\"weights\", {})}')
    m = bc.get('metrics', {})
    print('  Metrics:')
    print(f'    - avg_profile_alignment: {m.get(\"avg_profile_alignment\", \"N/A\")}')
    print(f'    - avg_ground_truth_quality: {m.get(\"avg_ground_truth_quality\", \"N/A\")}')
    print(f'    - avg_combined_score: {m.get(\"avg_combined_score\", \"N/A\")}')
except Exception as e:
    print(f'Error reading results: {e}')
" 2>/dev/null || echo "  (Install python3 to view summary)"
    else
        echo "⏳ Script still running... (results file is empty or incomplete)"
    fi
else
    echo "⏳ Script still running... (results file doesn't exist yet)"
fi

echo ""
echo "To view live logs:"
echo "  docker-compose logs -f api | grep -E '\[[0-9]+/9\]|Best configuration|Saved'"

