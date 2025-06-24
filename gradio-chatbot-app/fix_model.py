#!/usr/bin/env python3
"""
Script para solucionar problemas comunes con modelos de sentence-transformers
"""
import os
import sys
import shutil
from pathlib import Path

def clear_model_cache():
    """Limpia la caché de modelos de sentence-transformers"""
    print("🧹 Limpiando caché de modelos...")
    
    # Rutas comunes de caché
    cache_paths = [
        Path.home() / '.cache' / 'sentence_transformers',
        Path.home() / '.cache' / 'huggingface',
        Path.home() / '.cache' / 'torch' / 'sentence_transformers',
        Path.cwd() / 'sentence-transformers'
    ]
    
    for cache_path in cache_paths:
        if cache_path.exists():
            print(f"🗑️ Eliminando: {cache_path}")
            try:
                shutil.rmtree(cache_path)
                print(f"✅ Eliminado exitosamente")
            except Exception as e:
                print(f"⚠️ Error eliminando {cache_path}: {e}")
        else:
            print(f"ℹ️ No existe: {cache_path}")

def download_models_manually():
    """Descarga los modelos manualmente"""
    print("📥 Descargando modelos manualmente...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        models_to_try = [
            'all-MiniLM-L6-v2',
            'paraphrase-MiniLM-L6-v2',
            'all-mpnet-base-v2'
        ]
        
        for model_name in models_to_try:
            try:
                print(f"📦 Descargando {model_name}...")
                model = SentenceTransformer(model_name)
                print(f"✅ {model_name} descargado exitosamente")
                
                # Probar que funciona
                test_sentence = "This is a test sentence"
                embedding = model.encode(test_sentence)
                print(f"🧪 Prueba exitosa - dimensión: {len(embedding)}")
                
                return True
                
            except Exception as e:
                print(f"❌ Error con {model_name}: {str(e)}")
                continue
        
        return False
        
    except ImportError:
        print("❌ sentence-transformers no está instalado")
        return False

def reinstall_sentence_transformers():
    """Reinstala sentence-transformers"""
    print("🔄 Reinstalando sentence-transformers...")
    
    import subprocess
    
    try:
        # Desinstalar
        print("📤 Desinstalando sentence-transformers...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "sentence-transformers", "-y"])
        
        # Reinstalar
        print("📥 Reinstalando sentence-transformers...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "--no-cache-dir"])
        
        print("✅ Reinstalación completada")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en reinstalación: {e}")
        return False

def check_dependencies():
    """Verifica las dependencias necesarias"""
    print("🔍 Verificando dependencias...")
    
    required_packages = {
        'sentence_transformers': 'sentence-transformers',
        'torch': 'torch',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n📦 Instalando paquetes faltantes: {', '.join(missing_packages)}")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ Dependencias instaladas")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando dependencias: {e}")
            return False
    
    return True

def fix_permissions():
    """Corrige permisos en directorios de caché"""
    print("🔐 Verificando permisos...")
    
    cache_dirs = [
        Path.home() / '.cache',
        Path.home() / '.cache' / 'sentence_transformers',
        Path.home() / '.cache' / 'huggingface'
    ]
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            try:
                # Intentar crear un archivo de prueba
                test_file = cache_dir / 'test_permissions.tmp'
                test_file.touch()
                test_file.unlink()
                print(f"✅ Permisos OK: {cache_dir}")
            except Exception as e:
                print(f"⚠️ Problema de permisos en {cache_dir}: {e}")
                # Intentar corregir permisos (solo en sistemas Unix)
                if os.name != 'nt':
                    try:
                        os.chmod(cache_dir, 0o755)
                        print(f"🔧 Permisos corregidos: {cache_dir}")
                    except Exception as e2:
                        print(f"❌ No se pudieron corregir permisos: {e2}")

def test_model_loading():
    """Prueba la carga de modelos"""
    print("🧪 Probando carga de modelos...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Probar modelo más simple primero
        print("📝 Probando modelo básico...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Probar encoding
        test_texts = ["Hello world", "This is a test"]
        embeddings = model.encode(test_texts)
        
        print(f"✅ Modelo cargado exitosamente")
        print(f"📊 Dimensión de embeddings: {embeddings.shape}")
        print(f"🎯 Tipo de datos: {embeddings.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando modelo: {str(e)}")
        return False

def main():
    """Función principal para solucionar problemas"""
    print("🔧 SOLUCIONADOR DE PROBLEMAS DE MODELOS")
    print("=" * 50)
    
    # Paso 1: Verificar dependencias
    print("\n1️⃣ Verificando dependencias...")
    if not check_dependencies():
        print("❌ Hay problemas con las dependencias. Ejecuta:")
        print("pip install sentence-transformers torch numpy scikit-learn")
        return
    
    # Paso 2: Verificar permisos
    print("\n2️⃣ Verificando permisos...")
    fix_permissions()
    
    # Paso 3: Probar carga de modelo
    print("\n3️⃣ Probando carga de modelo...")
    if test_model_loading():
        print("\n🎉 ¡Todo funciona correctamente!")
        return
    
    # Paso 4: Limpiar caché si hay problemas
    print("\n4️⃣ Limpiando caché de modelos...")
    clear_model_cache()
    
    # Paso 5: Intentar descargar nuevamente
    print("\n5️⃣ Descargando modelos nuevamente...")
    if download_models_manually():
        print("\n🎉 ¡Problema resuelto!")
        return
    
    # Paso 6: Reinstalar como último recurso
    print("\n6️⃣ Reinstalando sentence-transformers...")
    if reinstall_sentence_transformers():
        print("\n7️⃣ Probando después de reinstalación...")
        if download_models_manually():
            print("\n🎉 ¡Problema resuelto después de reinstalación!")
            return
    
    # Si nada funciona
    print("\n❌ No se pudo resolver el problema automáticamente.")
    print("\n💡 Soluciones manuales:")
    print("1. Verificar conexión a internet")
    print("2. Ejecutar: pip install --upgrade sentence-transformers")
    print("3. Reiniciar el entorno Python")
    print("4. Verificar que no hay firewall bloqueando descargas")
    print("5. Probar en un entorno virtual nuevo")

