import gradio as gr
import logging
import os
import tempfile
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional
import re

# Librerías para RAG
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from dataclasses import dataclass

# Para usar Databricks endpoint
from model_serving_utils import query_endpoint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."

@dataclass
class DocumentChunk:
    """Representa un fragmento de documento con su contenido y metadatos"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class ZeroAgent:
    """Agente que determina si usar RAG o respuesta directa"""
    
    def __init__(self):
        # Patrones que indican necesidad de RAG (contenido específico)
        self.rag_keywords = [
            # MLOps/LLMOps específicos
            'mlops', 'llmops', 'devops', 'deployment', 'pipeline', 'ci/cd',
            'modelo', 'model', 'training', 'inference', 'monitoring',
            'databricks', 'kubernetes', 'docker', 'containerization',
            'feature store', 'model registry', 'experiment tracking',
            'hyperparameter', 'tuning', 'optimization', 'metrics',
            'evaluation', 'validation', 'testing', 'production',
            'mlflow', 'kubeflow', 'airflow', 'prefect',
            
            # Conceptos técnicos del dominio
            'arquitectura', 'framework', 'herramienta', 'tool',
            'best practice', 'mejores prácticas', 'patrón', 'pattern',
            'workflow', 'flujo de trabajo', 'automatización',
            'versionado', 'versioning', 'governance', 'gobernanza',
            
            # Preguntas sobre el libro específico
            'libro', 'book', 'capítulo', 'chapter', 'sección', 'section',
            'según el', 'de acuerdo al', 'menciona', 'explica',
            'big book', 'qué dice', 'cómo define'
        ]
        
        # Patrones que NO requieren RAG (conversación general)
        self.direct_patterns = [
            # Saludos y cortesías
            r'^(hola|hello|hi|hey|buenos días|buenas tardes|buenas noches)[\s\W]*$',
            r'^(¿cómo estás|how are you|cómo te va)[\?\s\W]*$',
            r'^(gracias|thank you|thanks|muchas gracias)[\s\W]*$',
            r'^(adiós|bye|goodbye|hasta luego|nos vemos)[\s\W]*$',
            
            # Preguntas sobre el sistema/bot
            r'^(¿qué eres|what are you|quién eres|who are you)',
            r'^(¿cómo funcionas|how do you work|qué puedes hacer)',
            r'^(¿en qué puedes ayudarme|what can you help)',
            
            # Respuestas muy cortas o vagas
            r'^(sí|no|ok|okay|bien|fine|perfecto)[\s\W]*$',
            r'^(test|testing|prueba)[\s\W]*$',
            
            # Conversación casual
            r'^(¿cómo te llamas|what\'s your name)',
            r'^(¿qué tal|how\'s it going)',
        ]
        
        # Palabras que indican consulta específica (fuerzan RAG)
        self.specific_query_indicators = [
            '¿qué es', '¿cómo se', '¿cuál es', '¿cuáles son',
            'explica', 'describe', 'define', 'compara',
            'diferencia', 'ejemplo', 'ventaja', 'desventaja',
            'implementa', 'proceso', 'paso', 'método',
            'what is', 'how to', 'explain', 'describe',
            'compare', 'difference', 'example', 'advantage'
        ]
    
    def should_use_rag(self, query: str) -> tuple[bool, str]:
        """
        Determina si usar RAG o respuesta directa
        Returns: (usar_rag: bool, razon: str)
        """
        query_lower = query.lower().strip()
        
        # 1. Verificar patrones de conversación directa (alta prioridad)
        for pattern in self.direct_patterns:
            if re.match(pattern, query_lower, re.IGNORECASE):
                return False, f"Patrón de conversación casual detectado: {pattern[:30]}..."
        
        # 2. Consultas muy cortas sin contenido específico
        if len(query.split()) <= 2 and not any(keyword in query_lower for keyword in self.rag_keywords):
            return False, "Consulta muy corta sin contenido técnico específico"
        
        # 3. Verificar indicadores de consulta específica (fuerzan RAG)
        for indicator in self.specific_query_indicators:
            if indicator in query_lower:
                return True, f"Indicador de consulta específica: '{indicator}'"
        
        # 4. Verificar keywords de dominio específico
        found_keywords = [kw for kw in self.rag_keywords if kw in query_lower]
        if found_keywords:
            return True, f"Keywords de dominio encontradas: {', '.join(found_keywords[:3])}"
        
        # 5. Consultas largas y complejas (probablemente técnicas)
        if len(query.split()) > 8:
            return True, "Consulta larga y compleja, probablemente requiere contexto específico"
        
        # 6. Default: usar conversación directa para consultas ambiguas
        return False, "Consulta general sin indicadores específicos de contenido técnico"

class RAGSystem:
    """Sistema RAG con Databricks Llama-4-Maverick y embeddings locales"""
    
    def __init__(self, default_pdf_path: str = None):
        self.embedding_model = None
        self.vector_store = None
        self.document_chunks: List[DocumentChunk] = []
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.default_pdf_path = default_pdf_path
        self.endpoint_name = os.getenv('SERVING_ENDPOINT')
        self.zero_agent = ZeroAgent()  # Inicializar el agente de routing
        
    def initialize_models(self):
        """Inicializa el modelo de embeddings (LLM ya está en Databricks)"""
        try:
            # Solo necesitamos modelo de embeddings local
            logger.info("Cargando modelo de embeddings...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info(f"Usando endpoint Databricks: {self.endpoint_name}")
            
            # Cargar PDF por defecto si está especificado
            if self.default_pdf_path and os.path.exists(self.default_pdf_path):
                logger.info(f"Cargando PDF por defecto: {self.default_pdf_path}")
                self.load_default_pdf()
            
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando sistema: {str(e)}")
            return False
    
    def extract_text_from_pdf(self, pdf_path: str) -> tuple[str, list]:
        """Extrae texto de un archivo PDF y mapea contenido a páginas"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                full_text = ""
                page_boundaries = []  # Lista de posiciones donde termina cada página
                current_position = 0
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text() + "\n"
                    full_text += page_text
                    current_position += len(page_text)
                    
                    page_boundaries.append({
                        'page_number': page_num,
                        'end_position': current_position,
                        'text_length': len(page_text)
                    })
                
                return full_text, page_boundaries
                
        except Exception as e:
            logger.error(f"Error extrayendo texto del PDF: {str(e)}")
            return "", []
    
    def chunk_text(self, text: str, filename: str, page_boundaries: list = None) -> List[DocumentChunk]:
        """Divide el texto en chunks más pequeños con mapeo preciso de páginas"""
        chunks = []
        words = text.split()
        
        # Función para obtener página real basada en posición
        def get_page_number(char_position: int) -> int:
            if not page_boundaries:
                # Fallback a estimación si no hay mapeo de páginas
                return (char_position // 2000) + 1  # ~2000 chars por página
            
            for page_info in page_boundaries:
                if char_position <= page_info['end_position']:
                    return page_info['page_number']
            return page_boundaries[-1]['page_number']  # Última página por defecto
        
        current_char_pos = 0
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_content = " ".join(chunk_words)
            
            # Calcular posición en caracteres del inicio del chunk
            words_before_chunk = words[:i]
            char_position_start = len(" ".join(words_before_chunk))
            char_position_end = char_position_start + len(chunk_content)
            
            # Obtener página real del chunk
            start_page = get_page_number(char_position_start)
            end_page = get_page_number(char_position_end)
            
            # Si el chunk abarca múltiples páginas, usar la página predominante
            chunk_page = start_page if (char_position_end - char_position_start) > 500 else end_page
            
            # Buscar títulos o secciones en el chunk para mejor contexto
            section_title = self._extract_section_title(chunk_content)
            
            chunk = DocumentChunk(
                content=chunk_content,
                metadata={
                    'filename': filename,
                    'chunk_index': len(chunks),
                    'word_count': len(chunk_words),
                    'start_word': i,
                    'end_word': min(i + self.chunk_size, len(words)),
                    'actual_page': chunk_page,  # Página real basada en PDF
                    'start_page': start_page,
                    'end_page': end_page,
                    'char_start': char_position_start,
                    'char_end': char_position_end,
                    'section_title': section_title,
                    'preview': chunk_content[:100] + "..." if len(chunk_content) > 100 else chunk_content
                }
            )
            chunks.append(chunk)
            
        return chunks
    
    def _extract_section_title(self, text: str) -> str:
        """Extrae el título de sección más probable del texto"""
        lines = text.split('\n')
        
        # Buscar patrones comunes de títulos
        for line in lines[:5]:  # Revisar primeras 5 líneas
            line = line.strip()
            
            # Títulos en mayúsculas
            if line.isupper() and 3 <= len(line) <= 80:
                return line
            
            # Títulos con números (ej: "1. Introduction", "Chapter 2")
            if any(pattern in line.lower() for pattern in ['chapter', 'section', 'part']):
                return line
            
            # Líneas que parecen títulos (cortas, sin puntos al final)
            if (10 <= len(line) <= 60 and 
                not line.endswith('.') and 
                not line.endswith(',') and
                line[0].isupper()):
                return line
        
        # Si no se encuentra título específico, usar las primeras palabras
        first_words = ' '.join(text.split()[:8])
        return first_words if len(first_words) < 80 else first_words[:77] + "..."
    
    def create_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Crea embeddings para los chunks"""
        if not self.embedding_model:
            logger.error("Modelo de embeddings no inicializado")
            return chunks
            
        try:
            contents = [chunk.content for chunk in chunks]
            embeddings = self.embedding_model.encode(contents)
            
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error creando embeddings: {str(e)}")
            return chunks
    
    def build_vector_store(self):
        """Construye el índice FAISS para búsqueda vectorial"""
        if not self.document_chunks:
            logger.warning("No hay chunks para indexar")
            return
            
        try:
            # Obtener dimensión de embeddings
            embedding_dim = len(self.document_chunks[0].embedding)
            
            # Crear índice FAISS
            self.vector_store = faiss.IndexFlatIP(embedding_dim)
            
            # Agregar embeddings al índice
            embeddings = np.array([chunk.embedding for chunk in self.document_chunks])
            # Normalizar embeddings para usar producto interno como similaridad coseno
            faiss.normalize_L2(embeddings)
            self.vector_store.add(embeddings)
            
            logger.info(f"Índice vectorial creado con {len(self.document_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error construyendo vector store: {str(e)}")
    
    def load_default_pdf(self) -> str:
        """Carga el PDF por defecto especificado en el constructor"""
        if not self.default_pdf_path or not os.path.exists(self.default_pdf_path):
            return "PDF por defecto no encontrado"
            
        try:
            # Extraer texto con mapeo de páginas
            logger.info(f"Procesando PDF por defecto: {self.default_pdf_path}")
            text, page_boundaries = self.extract_text_from_pdf(self.default_pdf_path)
            
            if not text.strip():
                return "No se pudo extraer texto del PDF por defecto"
            
            # Crear chunks con mapeo preciso de páginas
            filename = os.path.basename(self.default_pdf_path)
            chunks = self.chunk_text(text, filename, page_boundaries)
            logger.info(f"Creados {len(chunks)} chunks del PDF por defecto con mapeo de {len(page_boundaries)} páginas")
            
            # Crear embeddings
            chunks_with_embeddings = self.create_embeddings(chunks)
            
            # Agregar a la colección
            self.document_chunks.extend(chunks_with_embeddings)
            
            # Construir índice vectorial
            self.build_vector_store()
            
            return f"PDF por defecto cargado exitosamente: {filename}. {len(chunks)} chunks procesados con mapeo preciso de páginas."
            
        except Exception as e:
            logger.error(f"Error cargando PDF por defecto: {str(e)}")
            return f"Error cargando PDF por defecto: {str(e)}"
    
    def process_pdf(self, pdf_file) -> str:
        """Procesa un archivo PDF y actualiza la base de conocimientos"""
        if pdf_file is None:
            return "Por favor, selecciona un archivo PDF"
            
        try:
            # Guardar archivo temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
            
            # Extraer texto con mapeo de páginas
            logger.info(f"Procesando PDF: {pdf_file.name}")
            text, page_boundaries = self.extract_text_from_pdf(tmp_path)
            
            if not text.strip():
                return "No se pudo extraer texto del PDF"
            
            # Crear chunks con mapeo preciso de páginas
            chunks = self.chunk_text(text, pdf_file.name, page_boundaries)
            logger.info(f"Creados {len(chunks)} chunks con mapeo de {len(page_boundaries)} páginas")
            
            # Crear embeddings
            chunks_with_embeddings = self.create_embeddings(chunks)
            
            # Agregar a la colección existente
            self.document_chunks.extend(chunks_with_embeddings)
            
            # Reconstruir índice vectorial
            self.build_vector_store()
            
            # Limpiar archivo temporal
            os.unlink(tmp_path)
            
            return f"PDF procesado exitosamente. {len(chunks)} chunks agregados con mapeo preciso de páginas. Total: {len(self.document_chunks)} chunks en la base de conocimientos."
            
        except Exception as e:
            logger.error(f"Error procesando PDF: {str(e)}")
            return f"Error procesando PDF: {str(e)}"
    
    def search_similar_chunks(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        """Busca chunks similares a la consulta con scores de similaridad"""
        if not self.vector_store or not self.embedding_model:
            return []
            
        try:
            # Crear embedding de la consulta
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Buscar chunks similares
            scores, indices = self.vector_store.search(query_embedding, top_k)
            
            # Retornar chunks encontrados con scores
            similar_chunks = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.document_chunks):
                    chunk = self.document_chunks[idx]
                    # Agregar score de similaridad al metadata
                    chunk.metadata['similarity_score'] = float(scores[0][i])
                    similar_chunks.append(chunk)
            
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial: {str(e)}")
            return []
    
    def generate_direct_response(self, query: str, conversation_history: List = None) -> str:
        """Genera respuesta directa sin RAG para consultas generales"""
        try:
            # Prompt para conversación general sin contexto específico
            system_prompt = """Eres un asistente amigable y profesional especializado en MLOps y LLMOps. 

Responde de manera natural y conversacional. Para consultas generales, saludos o conversación casual, responde directamente sin buscar información específica.

Si la consulta requiere información técnica específica sobre MLOps/LLMOps, menciona que puedes ayudar con información detallada basada en documentación especializada."""

            # Preparar mensajes
            messages = []
            
            # Agregar historial si existe
            if conversation_history:
                messages.extend(conversation_history)
            
            messages.extend([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ])
            
            # Llamada a Databricks endpoint
            logger.info("Generando respuesta directa sin RAG")
            response = query_endpoint(
                endpoint_name=self.endpoint_name,
                messages=messages,
                max_tokens=300  # Respuestas más cortas para conversación general
            )
            
            return response.get("content", "No se pudo generar una respuesta")
            
        except Exception as e:
            logger.error(f"Error generando respuesta directa: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_response_with_databricks(self, query: str, context_chunks: List[DocumentChunk], conversation_history: List = None) -> tuple[str, list]:
        """Genera respuesta usando Databricks Llama-4-Maverick con contexto RAG"""
        try:
            # Construir contexto simple sin numeración (el LLM no necesita manejar referencias)
            context = "\n\n".join([chunk.content for chunk in context_chunks])
            
            # Crear el prompt sistema para Llama-4 (sin mencionar fuentes numeradas)
            system_prompt = """Eres un asistente experto en MLOps y LLMOps. Tu tarea es responder preguntas basándote ÚNICAMENTE en la información proporcionada en el contexto.

            INSTRUCCIONES IMPORTANTES:
            1. Usa SOLO la información del contexto proporcionado
            2. Si la información no está en el contexto, di que no tienes esa información específica
            3. Sé preciso, conciso y directo en tu respuesta
            4. Mantén un tono profesional pero accesible
            5. NO inventes información que no esté en el contexto"""

            user_prompt = f"""CONTEXTO:
            {context}

            PREGUNTA: {query}

            Por favor, responde la pregunta basándote únicamente en el contexto proporcionado."""

            # Preparar mensajes para Databricks endpoint
            messages = []
            
            # Agregar historial de conversación si existe
            if conversation_history:
                messages.extend(conversation_history)
            
            # Agregar prompt del sistema y consulta actual
            messages.extend([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            # Llamada a Databricks endpoint
            logger.info(f"Enviando consulta RAG a Databricks endpoint: {self.endpoint_name}")
            response = query_endpoint(
                endpoint_name=self.endpoint_name,
                messages=messages,
                max_tokens=800
            )
            response_content = response.get("content", "No se pudo generar una respuesta")
            
            # Crear referencias detalladas directamente del retrieval
            references = []
            for i, chunk in enumerate(context_chunks, 1):
                # Usar página real si está disponible, sino la estimada
                page_number = chunk.metadata.get('actual_page', chunk.metadata.get('estimated_page', 'N/A'))
                
                # Si abarca múltiples páginas, mostrar rango
                if chunk.metadata.get('start_page') and chunk.metadata.get('end_page'):
                    if chunk.metadata['start_page'] != chunk.metadata['end_page']:
                        page_display = f"{chunk.metadata['start_page']}-{chunk.metadata['end_page']}"
                    else:
                        page_display = str(chunk.metadata['start_page'])
                else:
                    page_display = str(page_number)
                
                reference = {
                    'source_number': i,
                    'filename': chunk.metadata['filename'],
                    'section_title': chunk.metadata.get('section_title', 'Sección no identificada'),
                    'page_number': page_display,
                    'similarity_score': chunk.metadata.get('similarity_score', 0.0),
                    'preview': chunk.metadata.get('preview', chunk.content[:100] + "..."),
                    'chunk_index': chunk.metadata.get('chunk_index', 'N/A'),
                    'content_length': len(chunk.content),
                    'char_range': f"{chunk.metadata.get('char_start', 'N/A')}-{chunk.metadata.get('char_end', 'N/A')}"
                }
                references.append(reference)
            
            return response_content, references
            
        except Exception as e:
            logger.error(f"Error generando respuesta con Databricks: {str(e)}")
            return f"Error consultando el modelo: {str(e)}", []

# Configuración del sistema
DEFAULT_PDF_PATH = "The Big Book of MLOps and LLMOps.pdf"

# Inicializar sistema RAG con PDF por defecto
rag_system = RAGSystem(default_pdf_path=DEFAULT_PDF_PATH)

def initialize_system():
    """Inicializa el sistema RAG"""
    success = rag_system.initialize_models()
    if success and rag_system.document_chunks:
        logger.info(f"Sistema inicializado con {len(rag_system.document_chunks)} chunks del PDF por defecto")
    return success

def process_pdf_upload(pdf_file):
    """Maneja la carga de archivos PDF"""
    return rag_system.process_pdf(pdf_file)

def query_rag_system(message, history):
    """Sistema inteligente que decide entre RAG o respuesta directa"""
    if not message.strip():
        return "ERROR: La pregunta no puede estar vacía"
    
    if not rag_system.document_chunks:
        return "Sistema no inicializado. Por favor, presiona 'Inicializar Sistema' primero."
    
    try:
        # ZERO AGENT: Decidir estrategia de respuesta
        use_rag, reason = rag_system.zero_agent.should_use_rag(message)
        
        logger.info(f"Zero Agent Decision: {'RAG' if use_rag else 'DIRECT'} - Razón: {reason}")
        
        # Convertir historial de Gradio a formato OpenAI para Databricks
        conversation_history = []
        if history:
            for user_msg, assistant_msg in history:
                conversation_history.append({"role": "user", "content": user_msg})
                if assistant_msg:  # El asistente podría no haber respondido aún
                    conversation_history.append({"role": "assistant", "content": assistant_msg})
        
        if use_rag:
            # RUTA RAG: Buscar contexto y generar respuesta con referencias
            similar_chunks = rag_system.search_similar_chunks(message, top_k=3)
            
            if not similar_chunks:
                return "No encontré información relevante en los documentos cargados."
            
            # Generar respuesta con RAG
            response, references = rag_system.generate_response_with_databricks(
                message, similar_chunks, conversation_history
            )
            
            # Construir respuesta final con referencias automáticas del retrieval
            final_response = response
            
            # Agregar sección de referencias (obtenidas directamente del retrieval)
            if references:
                final_response += "\n\n" + "="*50
                final_response += "\n📚 **FUENTES CONSULTADAS:**\n"
                
                for ref in references:
                    similarity_percentage = ref['similarity_score'] * 100
                    
                    final_response += f"\n**📄 Fuente {ref['source_number']}:**\n"
                    final_response += f"• **Archivo:** {ref['filename']}\n"
                    final_response += f"• **Sección:** {ref['section_title']}\n"
                    final_response += f"• **Página:** {ref['page_number']}\n"
                    final_response += f"• **Relevancia:** {similarity_percentage:.1f}%\n"
                    final_response += f"• **Vista previa:** {ref['preview']}\n"
            
            # Agregar nota sobre la decisión del agente (para transparencia)
            final_response += f"\n\n🤖 *Respuesta generada con RAG - {reason}*"
            
            return final_response
        
        else:
            # RUTA DIRECTA: Respuesta conversacional sin RAG
            response = rag_system.generate_direct_response(message, conversation_history)
            
            # Agregar nota sobre la decisión del agente (para transparencia)
            response += f"\n\n🤖 *Respuesta directa - {reason}*"
            
            return response
        
    except Exception as e:
        logger.error(f"Error en consulta del sistema: {str(e)}")
        return f"Error: {str(e)}"

def get_system_status():
    """Retorna el estado del sistema"""
    if not rag_system.embedding_model:
        return "❌ Sistema no inicializado"
    
    num_docs = len(set([chunk.metadata['filename'] for chunk in rag_system.document_chunks]))
    num_chunks = len(rag_system.document_chunks)
    
    # Mostrar información del PDF por defecto si está cargado
    default_pdf_loaded = "✅ PDF por defecto cargado" if any(
        chunk.metadata['filename'] == "The Big Book of MLOps and LLMOps.pdf" 
        for chunk in rag_system.document_chunks
    ) else "⚠️ PDF por defecto no cargado"
    
    endpoint_info = f"🚀 Endpoint: {rag_system.endpoint_name}"
    zero_agent_info = "🧠 Zero Agent: Activo"
    
    return f"✅ Sistema listo | {default_pdf_loaded} | {endpoint_info} | {zero_agent_info} | Documentos: {num_docs} | Chunks: {num_chunks}"

# Crear interfaz Gradio
with gr.Blocks(title="RAG con Databricks Llama-4 + Zero Agent - diseñado por Luiggi Silva") as demo:
    gr.Markdown("# 🤖 Sistema Multi Agente RAG con Databricks Llama-4-Maverick + Zero Agent - diseñado por Luiggi Silva")
    gr.Markdown("Sistema inteligente que **decide automáticamente** entre respuesta conversacional o búsqueda en documentos powered by **Databricks Llama-4-Maverick** y **'The Big Book of MLOps and LLMOps'**.")
    
    # Información del Zero Agent
    with gr.Accordion("🧠 ¿Qué es el Zero Agent?", open=False):
        gr.Markdown("""
        **Zero Agent** es un sistema inteligente que decide la mejor estrategia para responder tu consulta:
        
        - **Respuesta Directa**: Para saludos, conversación casual, preguntas generales sobre el sistema
        - **RAG (Retrieval)**: Para consultas técnicas específicas sobre MLOps/LLMOps que requieren contexto del libro
        
        **Ejemplos:**
        - "Hola" → Respuesta directa
        - "¿Qué es MLOps?" → RAG con referencias del libro
        - "Gracias" → Respuesta directa
        - "Explica CI/CD en ML" → RAG con contexto específico
        """)
    
    # Estado del sistema
    status_text = gr.Textbox(
        label="Estado del Sistema",
        value="Presiona 'Inicializar Sistema' para cargar el libro y configurar Databricks...",
        interactive=False
    )
    
    # Botón de inicialización
    init_btn = gr.Button("🚀 Inicializar Sistema", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Información del documento preconfigurado
            gr.Markdown("## 📚 Documento precargado")
            gr.Markdown("**The Big Book of MLOps and LLMOps**")
            gr.Markdown("- Conceptos de MLOps")
            gr.Markdown("- Prácticas de LLMOps") 
            gr.Markdown("- Herramientas y frameworks")
            gr.Markdown("- Casos de uso y ejemplos")
            
            # Información del modelo y Zero Agent
            gr.Markdown("## 🧠 AI System")
            gr.Markdown("**Databricks Llama-4-Maverick + Zero Agent**")
            gr.Markdown("- Routing inteligente de consultas")
            gr.Markdown("- Respuestas contextuales precisas")
            gr.Markdown("- Referencias automáticas cuando necesario")
            gr.Markdown("- Conversación natural para chat casual")
            
        
        with gr.Column(scale=2):
            # Chat interface
            gr.Markdown("## 💬 Chat Inteligente con Zero Agent")
            chatbot = gr.Chatbot(
                height=400,
                label="Conversación con routing automático"
            )
            msg = gr.Textbox(
                label="Tu mensaje (el sistema decidirá automáticamente la mejor respuesta)",
                placeholder="Ejemplo: 'Hola' o '¿Qué diferencias hay entre MLOps y LLMOps?'",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("📤 Enviar", variant="primary")
                clear_btn = gr.Button("🗑️ Limpiar")
    
    # Ejemplos para mostrar el Zero Agent en acción
    gr.Markdown("## 💡 Ejemplos para probar el Zero Agent")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🗣️ Conversación Casual (Respuesta Directa)")
            casual_examples = gr.Examples(
                examples=[
                    "Hola",
                    "¿Cómo estás?",
                    "Gracias",
                    "¿Qué puedes hacer?",
                    "Buenos días",
                    "Test"
                ],
                inputs=msg,
                label="Prueba estos para ver respuestas directas"
            )
        
        with gr.Column():
            gr.Markdown("### 🔍 Consultas Técnicas (RAG con Referencias)")
            technical_examples = gr.Examples(
                examples=[
                    "¿Qué es MLOps y cuáles son sus beneficios principales?",
                    "Explica las diferencias entre MLOps y LLMOps",
                    "¿Cuáles son las herramientas más importantes para MLOps?",
                    "¿Cómo se implementa CI/CD en proyectos de ML?",
                    "Describe el ciclo de vida completo de un modelo ML",
                    "¿Qué métricas son importantes para monitoreo?"
                ],
                inputs=msg,
                label="Prueba estos para ver RAG con fuentes"
            )
    
    # Event handlers
    def init_system():
        if initialize_system():
            status = get_system_status()
            if "PDF por defecto cargado" in status:
                return f"✅ Sistema inicializado correctamente con Databricks Llama-4-Maverick y Zero Agent.\n'The Big Book of MLOps and LLMOps' está listo para consultar.\n🧠 El Zero Agent decidirá automáticamente entre respuesta directa o RAG según tu consulta.\n💡 Las consultas técnicas incluirán referencias detalladas, mientras que la conversación casual será natural."
            else:
                return f"⚠️ Sistema inicializado pero no se pudo cargar el PDF por defecto. Verifica que el archivo existe en: {DEFAULT_PDF_PATH}"
        else:
            return "❌ Error inicializando sistema. Verifica la configuración de SERVING_ENDPOINT."
    
    def update_status():
        return get_system_status()
    
    def respond(message, history):
        response = query_rag_system(message, history)
        history.append([message, response])
        return "", history
    
    def clear_chat():
        return None, ""
    
    # Conectar eventos
    init_btn.click(init_system, outputs=status_text)    
    submit_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear_btn.click(clear_chat, outputs=[chatbot, msg])
    
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

if __name__ == "__main__":
    import fix_model
    fix_model.main()
    demo.launch(share=True)