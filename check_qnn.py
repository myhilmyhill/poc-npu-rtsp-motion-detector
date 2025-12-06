"""
Quick script to verify QNN Execution Provider is available
"""
import onnxruntime as ort

print("=" * 60)
print("ONNX Runtime QNN Status Check")
print("=" * 60)

print(f"\n📦 ONNX Runtime Location:")
print(f"   {ort.__file__}")

print(f"\n🔌 Available Execution Providers:")
providers = ort.get_available_providers()
for provider in providers:
    emoji = "🚀" if provider == "QNNExecutionProvider" else "  "
    print(f"   {emoji} {provider}")

if 'QNNExecutionProvider' in providers:
    print(f"\n✅ SUCCESS: QNN Execution Provider is available!")
    print(f"   Your NPU hardware acceleration is ready to use.")
else:
    print(f"\n❌ WARNING: QNN Execution Provider is NOT available.")
    print(f"   Inference will fall back to CPU.")

# Try creating a session with QNN
print(f"\n🧪 Testing QNN Session Creation...")
try:
    from pathlib import Path
    model_path = "models/yolov8n.onnx"
    
    if not Path(model_path).exists():
        print(f"   ⚠️  Model not found: {model_path}")
        print(f"   Skipping session creation test.")
    else:
        qnn_options = {
            'backend_path': 'QnnHtp.dll',
            'qnn_context_priority': 'high',
        }
        
        session = ort.InferenceSession(
            model_path,
            providers=[
                ('QNNExecutionProvider', qnn_options),
                'CPUExecutionProvider'
            ]
        )
        
        active_providers = session.get_providers()
        print(f"   Active Providers: {active_providers}")
        
        if 'QNNExecutionProvider' in active_providers:
            print(f"   ✅ QNN session created successfully!")
        else:
            print(f"   ⚠️  QNN provider requested but session is using: {active_providers[0]}")
            
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
