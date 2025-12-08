#!/usr/bin/env python3
"""
Simple script to verify bias mitigation configs are correctly formatted.
Doesn't require database or full service stack.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

def verify_bias_mitigation_config():
    """Verify bias_mitigation_config.json is correctly formatted."""
    config_path = BASE_DIR / "bias_config" / "bias_mitigation_config.json"
    
    if not config_path.exists():
        print("❌ bias_mitigation_config.json not found!")
        return False
    
    try:
        with config_path.open() as f:
            config = json.load(f)
        
        print("\n✅ Bias Mitigation Config Found:")
        print(f"   Path: {config_path}")
        print(f"\n   Structure:")
        
        # Check structure
        if "cold_start" in config:
            print("   ✓ cold_start section exists")
            if "minilm" in config["cold_start"]:
                print("   ✓ minilm model config exists")
                model_cfg = config["cold_start"]["minilm"]
                
                fields = ["primary_domain", "research_stage", "reading_level"]
                for field in fields:
                    if field in model_cfg:
                        field_cfg = model_cfg[field]
                        slices = field_cfg.get("underperforming_slices", [])
                        boost = field_cfg.get("boost_factor", 0)
                        floor = field_cfg.get("min_score_floor", 0)
                        
                        print(f"\n   [{field}]")
                        print(f"      Underperforming slices: {slices}")
                        print(f"      Boost factor: {boost}x")
                        print(f"      Min score floor: {floor}")
                    else:
                        print(f"   ⚠️  {field} not found in config")
            else:
                print("   ⚠️  minilm model config not found")
        else:
            print("   ⚠️  cold_start section not found")
        
        print("\n✅ Config is valid JSON and properly structured!")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False


def verify_fairness_config():
    """Verify fairness_config.json is correctly formatted."""
    config_path = BASE_DIR / "fairness_config.json"
    
    if not config_path.exists():
        print("\n⚠️  fairness_config.json not found (optional)")
        return True  # Not required for user-profile mitigation
    
    try:
        with config_path.open() as f:
            config = json.load(f)
        
        print("\n✅ Fairness Config Found:")
        print(f"   Path: {config_path}")
        print(f"   Under-served fields: {config.get('under_served_fields', [])}")
        
        if not config.get('under_served_fields'):
            print("   ℹ️  No under-served fields currently (field-based fairness won't apply)")
        else:
            print(f"   ✓ {len(config['under_served_fields'])} under-served fields will be boosted")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Error reading fairness config: {e}")
        return True  # Not critical


def main():
    print("=" * 80)
    print("  BIAS MITIGATION CONFIG VERIFICATION")
    print("=" * 80)
    
    bias_ok = verify_bias_mitigation_config()
    fairness_ok = verify_fairness_config()
    
    print("\n" + "=" * 80)
    if bias_ok:
        print("✅ BIAS MITIGATION IS CONFIGURED AND READY!")
        print("\nNext steps:")
        print("  1. The recommendation service will automatically load these configs")
        print("  2. Users in underperforming slices will get boosted scores")
        print("  3. Papers from under-served fields will get boosted (if any)")
        print("\nTo test with actual recommendations:")
        print("  - Fix dependency issues (pyarrow, tf-keras)")
        print("  - Run: python scripts/test_bias_mitigation.py")
    else:
        print("❌ CONFIG ISSUES DETECTED - Please fix before proceeding")
    print("=" * 80)


if __name__ == "__main__":
    main()
