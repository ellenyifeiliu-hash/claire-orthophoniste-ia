import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any
import replicate
from PIL import Image
import io
import base64
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

from langchain_groq import ChatGroq
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

import gradio as gr
from elevenlabs.client import ElevenLabs  # CORRIGÉ : Import pour v2
import dotenv

dotenv.load_dotenv()

# ================== Clés API ==================
CLE_GROQ = os.getenv("CLE_GROQ")
CLE_ELEVENLABS = os.getenv("CLE_ELEVENLABS")
JETON_REPLICATE = os.getenv("JETON_REPLICATE")

os.environ["REPLICATE_API_TOKEN"] = JETON_REPLICATE

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.6, api_key=CLE_GROQ)

ID_VOIX = "21m00Tcm4TlvDq8ikWAM"  # Rachel

# Client ElevenLabs (CORRIGÉ pour v2)
client_eleven = ElevenLabs(api_key=CLE_ELEVENLABS)

# ================== Base de Données ==================
BASE = "claire_pro.db"

def initialiser_base():
    conn = sqlite3.connect(BASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS enfants (
        nom TEXT PRIMARY KEY, donnees TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS seances (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nom TEXT, horodatage TEXT, categorie TEXT, mot TEXT, 
        reaction TEXT, image_b64 TEXT, chemin_audio TEXT)''')
    conn.commit()
    conn.close()

initialiser_base()

# ================== Génération d'Images ==================
def generer_image(mot: str) -> str:
    invite = f"photo réaliste très douce et lumineuse pour enfant autiste 2-4 ans, un seul {mot} au centre, fond blanc ou pastel très clair, style livre cartonné Petit Ours Brun, lumière chaude, hyper détaillé, pas de texte, pas de bordure, qualité maximale"
    
    sortie = replicate.run(
        "black-forest-labs/flux-pro",
        input={"prompt": invite, "num_outputs": 1, "aspect_ratio": "1:1", "output_format": "png"}
    )
    url_img = sortie[0]
    
    import requests
    donnees_img = requests.get(url_img).content
    b64 = base64.b64encode(donnees_img).decode()
    return f"data:image/png;base64,{b64}"

# ================== RAG (Vocabulaire) ==================
try:
    lecteur = SimpleDirectoryReader("data", required_exts=[".docx"])
    documents = lecteur.load_data()
    client_chroma = chromadb.PersistentClient(path="chroma_db_pro")
    collection = client_chroma.get_or_create_collection("vocabulaire")
    magasin_vecteurs = ChromaVectorStore(chroma_collection=collection)
    contexte_stockage = StorageContext.from_defaults(vector_store=magasin_vecteurs)
    index = VectorStoreIndex.from_documents(documents, contexte_stockage=contexte_stockage)
except Exception as e:
    print(f"Attention : Pas de fichiers DOCX dans data/. RAG désactivé. Erreur: {e}")
    index = None  # Fallback sans RAG

# ================== Invite Système ==================
INVITE_SYSTEME = """
Tu t'appelles Claire, orthophoniste spécialisée autisme à Paris.
Tu parles exclusivement en français, voix extrêmement douce et lente.
Âge langagier de l'enfant : {age_lang}.
Règles d'or :
- Maximum 6 mots par phrase
- Répéter chaque mot cible 3 fois
- Pauses très longues : ………
- Encouragement énorme à la moindre micro-réaction
- Tu ne demandes jamais "répète", tu modélises seulement
Commence toujours par : "Bonjour mon petit cœur [prénom]… c'est Claire……"
Quand tu dis un mot, écris-le en MAJUSCULES pour que le système génère l'image.
Exemple : Regarde…… CHAT…… chat…… chat…… il dort…… miam miam……
"""

# ================== Dialogue Principal ==================
def discuter_avec_claire(message: str, historique: List, etat_enfant: dict, categorie: str):
    etat = json.loads(etat_enfant) if isinstance(etat_enfant, str) else {}
    nom = etat.get("nom", "Léo")
    
    # RAG ou fallback
    if index:
        reponse = index.as_query_engine(similarity_top_k=12).query(
            f"Liste 8 mots très simples en français de la catégorie '{categorie}' pour âge langagier {etat.get('age_langage','2 ans')}. Retourne seulement les mots séparés par virgule."
        )
        mots_du_jour = [m.strip().lower() for m in reponse.response.split(",")[:5]]
    else:
        # Fallback sans DOCX
        mots_du_jour = ["chat", "chien", "pomme", "eau", "maman"] if categorie == "animaux familiers" else ["maman", "papa", "bébé"]
    
    invite = INVITE_SYSTEME.format(age_lang=etat.get("age_langage", "2 ans"))
    complet = f"""{invite}

Prénom : {nom}
Catégorie : {categorie}
Mots du jour : {', '.join(mots_du_jour)}

Historique récent : {historique[-8:]}

L'enfant dit ou fait : '{message or '(silence ou regarde ailleurs)'}'

Réponds exactement comme Claire le ferait, avec majuscules sur les mots cibles.
"""
    reponse_claire = llm.invoke(complet).content.strip()
    
    # Images
    images = []
    for mot in mots_du_jour:
        if mot.upper() in reponse_claire:
            img_b64 = generer_image(mot)
            images.append(img_b64)
            
            # Enregistrer
            conn = sqlite3.connect(BASE)
            c = conn.cursor()
            c.execute("INSERT INTO seances (nom,horodatage,categorie,mot,reaction,image_b64) VALUES (?,?,?,?,?,?)",
                     (nom, datetime.now().isoformat(), categorie, mot, message or "regarde écran", img_b64))
            conn.commit()
            conn.close()
    
    # Audio CORRIGÉ v2
    try:
        audio_gen = client_eleven.text_to_speech.convert(
            text=reponse_claire,
            voice_id=ID_VOIX,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128"
        )
        
        chemin_audio = "temp_audio.mp3"
        with open(chemin_audio, "wb") as f:
            for chunk in audio_gen:
                f.write(chunk)
    except Exception as e:
        print(f"Erreur audio (clé API ?): {e}")
        chemin_audio = None  # Fallback sans audio
    
    return reponse_claire, chemin_audio, images if images else None

# ================== Rapport Hebdomadaire ==================
def generer_rapport_semaine(nom: str):
    conn = sqlite3.connect(BASE)
    c = conn.cursor()
    lundi = (datetime.now() - timedelta(days=datetime.now().weekday())).strftime("%Y-%m-%d")
    c.execute("SELECT mot, COUNT(*) as nb, reaction FROM seances WHERE nom=? AND horodatage >= ? GROUP BY mot ORDER BY nb DESC", (nom, lundi))
    donnees = c.fetchall()
    conn.close()
    
    if not donnees:
        return "Aucune séance cette semaine."
    
    tampon = io.BytesIO()
    doc = SimpleDocTemplate(tampon, pagesize=A4)
    styles = getSampleStyleSheet()
    histoire = []
    
    histoire.append(Paragraph(f"<font size=18>Rapport hebdomadaire – {nom}</font>", styles['Title']))
    histoire.append(Spacer(1, 20))
    histoire.append(Paragraph(f"<b>Semaine du {lundi}</b>", styles['Normal']))
    histoire.append(Spacer(1, 30))
    
    for mot, count, reaction in donnees[:10]:
        histoire.append(Paragraph(f"• <b>{mot.upper()}</b> présenté {count} fois", styles['Normal']))
        histoire.append(Paragraph(f"    Réactions : {reaction}", styles['Normal']))
        histoire.append(Spacer(1, 12))
    
    histoire.append(PageBreak())
    histoire.append(Paragraph("Photos :", styles['Heading2']))
    
    conn = sqlite3.connect(BASE)
    c = conn.cursor()
    c.execute("SELECT image_b64 FROM seances WHERE nom=? AND horodatage >= ? LIMIT 20", (nom, lundi))
    for ligne in c.fetchall():
        if ligne[0]:
            donnees_img = base64.b64decode(ligne[0].split(",")[1])
            img = RLImage(io.BytesIO(donnees_img), width=10*cm, height=10*cm)
            histoire.append(img)
            histoire.append(Spacer(1, 10))
    conn.close()
    
    doc.build(histoire)
    tampon.seek(0)
    return tampon

# ================== Interface Gradio ==================
with gr.Blocks(title="Claire Pro Max", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦋 Claire Pro Max\n**Orthophoniste virtuelle pour enfants autistes**")
    
    with gr.Row():
        saisie_nom = gr.Textbox(label="Prénom de l'enfant", value="Léo")
        categorie = gr.Dropdown(
            choices=["famille", "animaux familiers", "nourriture", "actions", "objets & jouets", "émotions", "couleurs", "corps"],
            value="animaux familiers", label="Catégorie"
        )
    
    chatbot = gr.Chatbot(height=500, avatar_images=["🧑‍⚕️", "👶"])
    saisie_msg = gr.Textbox(placeholder="Message de l'enfant (ou vide pour silence)", label="Observation")
    
    with gr.Row():
        audio = gr.Audio(label="Voix de Claire", autoplay=True)
        galerie = gr.Gallery(label="Images générées", height=400)
    
    btn_rapport = gr.Button("📄 Rapport PDF", variant="primary")
    sortie_rapport = gr.File(label="Télécharger")
    
    def mettre_a_jour_etat(nom):
        conn = sqlite3.connect(BASE)
        c = conn.cursor()
        c.execute("SELECT donnees FROM enfants WHERE nom=?", (nom,))
        ligne = c.fetchone()
        conn.close()
        if ligne:
            return json.dumps(json.loads(ligne[0]))
        return json.dumps({"nom": nom, "age_langage": "2 ans", "mots_maitrises": ["maman","papa"]})
    
    def envoyer(message, historique, json_etat, cat):
        etat = json.loads(json_etat)
        reponse, chemin_audio, images = discuter_avec_claire(message, historique, etat, cat)
        historique.append(("👶 " + (message or "(silence)"), None))
        historique.append(("🧑‍⚕️ " + reponse, images[0] if images else None))
        return historique, chemin_audio, images or [], ""
    
    saisie_msg.submit(envoyer, [saisie_msg, chatbot, gr.State(value=""), categorie], [chatbot, audio, galerie, saisie_msg])
    
    btn_rapport.click(generer_rapport_semaine, saisie_nom, sortie_rapport)

if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)
