#!/usr/bin/env python3
"""List all available Gemini models using the new SDK."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

def list_all_available_models():
    """List all available Gemini models from the API."""
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    try:
        # Create client
        print("🔗 Creating client...")
        client = genai.Client(api_key=api_key)
        
        print("🔍 FETCHING ALL AVAILABLE MODELS...")
        print("=" * 80)
        
        # List all models using the SDK
        print("📡 Calling API...")
        models = list(client.models.list())
        print(f"📡 API returned {len(models) if models else 0} models")
        
        if not models:
            print("❌ No models returned from API")
            return False
        
        print(f"\n📊 FOUND {len(models)} AVAILABLE MODELS:")
        print("=" * 80)
        
        # Sort models by name for better readability
        models_sorted = sorted(models, key=lambda m: m.name)
        
        # Group models by family
        model_families = {}
        
        for model in models_sorted:
            # Extract model family (e.g., "gemini-2.5" from "models/gemini-2.5-pro")
            model_name = model.name.replace("models/", "")
            
            # Skip non-generative models
            if any(skip in model_name.lower() for skip in ['embedding', 'aqa', 'text-']):
                continue
                
            family = "-".join(model_name.split("-")[:2])  # e.g., "gemini-2.5"
            
            if family not in model_families:
                model_families[family] = []
            
            model_families[family].append({
                'name': model_name,
                'full_name': model.name,
                'display_name': getattr(model, 'display_name', 'N/A'),
                'description': (getattr(model, 'description', 'N/A') or 'N/A')[:100] + "..." if hasattr(model, 'description') and getattr(model, 'description') and len(getattr(model, 'description', '')) > 100 else (getattr(model, 'description', 'N/A') or 'N/A'),
                'input_limit': getattr(model, 'input_token_limit', 'N/A'),
                'output_limit': getattr(model, 'output_token_limit', 'N/A')
            })
        
        # Display models by family
        for family, family_models in model_families.items():
            print(f"\n🚀 {family.upper()} FAMILY:")
            print("-" * 60)
            
            for model_info in family_models:
                print(f"  📋 {model_info['name']}")
                print(f"     Display: {model_info['display_name']}")
                print(f"     Input Tokens: {model_info['input_limit']:,}" if isinstance(model_info['input_limit'], int) else f"     Input Tokens: {model_info['input_limit']}")
                print(f"     Output Tokens: {model_info['output_limit']:,}" if isinstance(model_info['output_limit'], int) else f"     Output Tokens: {model_info['output_limit']}")
                if model_info['description'] != 'N/A':
                    print(f"     Description: {model_info['description']}")
                print()
        
        # Show just the model names for easy copying
        print("\n📝 MODEL NAMES FOR CONFIG:")
        print("=" * 40)
        all_model_names = []
        for family_models in model_families.values():
            for model_info in family_models:
                all_model_names.append(model_info['name'])
        
        for name in sorted(all_model_names):
            print(f"  {name}")
        
        # Test a few promising models
        print(f"\n🧪 TESTING PROMISING MODELS:")
        print("=" * 40)
        
        test_models = [name for name in all_model_names if any(x in name for x in ['2.5-pro', '2.0-flash'])]
        
        for model_name in test_models[:5]:  # Test up to 5 models
            try:
                print(f"\n🔍 Testing {model_name}...")
                response = client.models.generate_content(
                    model=model_name,
                    contents="Say 'working' if you can respond",
                    config=genai.types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=10
                    )
                )
                
                if response.text and "working" in response.text.lower():
                    print(f"  ✅ {model_name} - WORKING")
                else:
                    print(f"  ⚠️  {model_name} - Response: '{response.text[:30]}...'")
                    
            except Exception as e:
                print(f"  ❌ {model_name} - ERROR: {str(e)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False

if __name__ == "__main__":
    print("🤖 GEMINI MODELS DISCOVERY")
    print("=" * 50)
    
    list_all_available_models()