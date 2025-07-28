#!/usr/bin/env python3
"""
Test script for the Neuromorphic Dream Engine
Tests the complete workflow: separation -> training -> generation -> track creation
"""

import asyncio
import os
import sys
from pathlib import Path



from src.services.separation_service import SeparationService
from src.services.training_service import TrainingService
from src.services.generative_service import GenerativeService
from src.database.service import DatabaseService


async def test_neuromorphic_engine():
    """
    Complete test of the Neuromorphic Dream Engine workflow
    """
    print("🧠 Starting Neuromorphic Dream Engine Test...")
    
    # Initialize services
    db_service = DatabaseService()
    separation_service = SeparationService()
    training_service = TrainingService()
    generative_service = GenerativeService()
    
    try:
        # Test 1: Database initialization
        print("\n📊 Testing database initialization...")
        stats = await db_service.get_stem_statistics()
        print(f"Current stems in database: {stats['total_stems']}")
        
        # Test 2: Check if we have any stereo tracks for analysis
        stereo_tracks_dir = Path("stereo_tracks_for_analysis")
        if not stereo_tracks_dir.exists():
            print(f"⚠️  Directory {stereo_tracks_dir} does not exist")
            return
        
        audio_files = list(stereo_tracks_dir.glob("*.wav")) + list(stereo_tracks_dir.glob("*.mp3"))
        if not audio_files:
            print("⚠️  No audio files found in stereo_tracks_for_analysis/")
            print("   Please add some .wav or .mp3 files to test separation")
            return
        
        # Test 3: Audio separation
        print(f"\n🎵 Testing audio separation with {len(audio_files)} files...")
        test_file = audio_files[0]
        print(f"Separating: {test_file.name}")
        
        try:
            separated_stems = await separation_service.separate_track_async(str(test_file))
            print(f"✅ Separation successful! Generated {len(separated_stems)} stems")
            for stem_type, stem_path in separated_stems.items():
                print(f"   - {stem_type}: {Path(stem_path).name}")
        except Exception as e:
            print(f"❌ Separation failed: {e}")
            return
        
        # Test 4: Check available categories for training
        print("\n🎯 Checking available categories for training...")
        categories = await db_service.get_stem_categories()
        if not categories:
            print("⚠️  No categories found in database")
            print("   You may need to process some stems first")
            return
        
        print(f"Available categories: {categories}")
        
        # Test 5: VAE Training (on first available category)
        test_category = categories[0]
        print(f"\n🧠 Testing VAE training for category: {test_category}")
        
        try:
            model_path = await training_service.train_vae_async(test_category)
            if model_path:
                print(f"✅ Training successful! Model saved to: {model_path}")
            else:
                print("❌ Training failed or insufficient data")
                return
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return
        
        # Test 6: Generative stem creation
        print(f"\n✨ Testing generative stem creation for category: {test_category}")
        
        try:
            generated_stems = await generative_service.generate_stems_async(
                category=test_category,
                num_variations=3,
                mode="random"
            )
            if generated_stems:
                print(f"✅ Generation successful! Created {len(generated_stems)} new stems")
                for stem_path in generated_stems:
                    print(f"   - {Path(stem_path).name}")
            else:
                print("❌ Generation failed")
                return
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return
        
        # Test 7: Final statistics
        print("\n📈 Final statistics:")
        final_stats = await db_service.get_stem_statistics()
        print(f"Total stems: {final_stats['total_stems']}")
        for source_stat in final_stats['by_source']:
            print(f"   - {source_stat['source']}: {source_stat['count']} stems")
        
        print("\n🎉 Neuromorphic Dream Engine test completed successfully!")
        print("\n🚀 The system is ready for:")
        print("   - Audio separation with Demucs")
        print("   - VAE training on stem categories")
        print("   - Generative stem creation")
        print("   - Complete track generation workflow")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        db_service.cleanup()
        separation_service.cleanup()


async def test_api_endpoints():
    """
    Test the API endpoints (requires server to be running)
    """
    print("\n🌐 Testing API endpoints...")
    
    try:
        import httpx
        
        base_url = "http://localhost:8000"
        
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✅ Health endpoint working")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return
            
            # Test stems endpoint
            response = await client.get(f"{base_url}/api/v1/stems/")
            if response.status_code == 200:
                stems_data = response.json()
                print(f"✅ Stems endpoint working - found {len(stems_data.get('stems', []))} stems")
            else:
                print(f"❌ Stems endpoint failed: {response.status_code}")
            
            # Test neuromorphic endpoints
            endpoints_to_test = [
                "/api/v1/neuromorphic/preprocess",
                "/api/v1/neuromorphic/train", 
                "/api/v1/neuromorphic/generate"
            ]
            
            for endpoint in endpoints_to_test:
                try:
                    # Just check if endpoint exists (POST without data will return 422)
                    response = await client.post(f"{base_url}{endpoint}")
                    if response.status_code in [422, 400]:  # Expected for missing data
                        print(f"✅ {endpoint} endpoint available")
                    else:
                        print(f"⚠️  {endpoint} returned unexpected status: {response.status_code}")
                except Exception as e:
                    print(f"❌ {endpoint} failed: {e}")
    
    except ImportError:
        print("⚠️  httpx not available - skipping API tests")
        print("   Install with: pip install httpx")
    except Exception as e:
        print(f"❌ API test failed: {e}")


if __name__ == "__main__":
    print("🧠 Neuromorphic Dream Engine - Complete System Test")
    print("=" * 60)
    
    # Run core engine tests
    asyncio.run(test_neuromorphic_engine())
    
    # Ask user if they want to test API endpoints
    print("\n" + "=" * 60)
    test_api = input("\n🌐 Test API endpoints? (requires server running) [y/N]: ")
    if test_api.lower() in ['y', 'yes']:
        print("\n📡 Make sure the server is running with: python src/main.py")
        input("Press Enter when ready...")
        asyncio.run(test_api_endpoints())
    
    print("\n✨ Test completed!")