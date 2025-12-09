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
from elevenlabs.client import ElevenLabs  # CORRIGÉ : Nouvelle API v2
import dotenv

dotenv.load_dotenv()

# ================== Clés API ==================
CLE_GROQ = os.getenv("CLE_GROQ")
CLE_ELEVENLABS = os.getenv("CLE_ELEVENLABS")
JETON_REPLICATE = os.getenv("JETON_REPLICATE")  # Obligatoire pour Flux

os.environ["REPLICATE_API_TOKEN"] = JETON_REPLICATE

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.6, api_key=CLE_GROQ)

ID_VOIX = "21m00Tcm4TlvDq8ikWAM"  # Rachel, la voix préférée des enfants

# Client ElevenLabs global (CORRIGÉ)
client_eleven = ElevenLabs(api_key=CLE_ELEVENLABS)

# ================== Base de Données + Enregistrements ==================
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

# ================== Génération d'Images Flux (Style Réaliste Ultra-Doux) ==================
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

# ================== RAG ==================
lecteur = SimpleDirectoryReader("data", required_exts=[".docx"])
documents = lecteur.load_data()
client_chroma = chromadb.PersistentClient(path="chroma_db_pro")
collection = client_chroma.get_or_create_collection("vocabulaire")
magasin_vecteurs = ChromaVectorStore(chroma_collection=collection)
contexte_stockage = StorageContext.from_defaults(vector_store=magasin_vecteurs)
index = VectorStoreIndex.from_documents(documents, contexte_stockage=contexte_stockage)

# ================== Invite Système (Version Ultime) ==================
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

# ================== Fonction de Dialogue Principale ==================
def discuter_avec_claire(message: str, historique: List, etat_enfant: dict, categorie: str):
    etat = json.loads(etat_enfant) if isinstance(etat_enfant, str) else {}
    nom = etat.get("nom", "Léo")
    
    # RAG : Récupérer vocabulaire du jour
    reponse = index.as_query_engine(similarity_top_k=12).query(
        f"Liste 8 mots très simples en français de la catégorie '{categorie}' pour âge langagier {etat.get('age_langage','2 ans')}. Retourne seulement les mots séparés par virgule."
    )
    mots_du_jour = [m.strip().lower() for m in reponse.response.split(",")[:5]]
    
    invite = INVITE_SYSTEME.format(age_lang=etat.get("age_langage", "2 ans"))
    complet = f"""{invite}

Prénom : {nom}
Catégorie : {categorie}
Mots du jour : {", ".join(mots_du_jour)}

Historique récent : {historique[-8:]}

L'enfant dit ou fait : "{message or '（silence ou regarde ailleurs）'}"

Réponds exactement comme Claire le ferait, avec majuscules sur les mots cibles pour déclencher l'image.
"""
    reponse_claire = llm.invoke(complet).content.strip()
    
    # Génération automatique d'images
    images = []
    for mot in mots_du_jour:
        if mot.upper() in reponse_claire:
            img_b64 = generer_image(mot)
            images.append(img_b64)
            
            # Enregistrer la séance
            conn = sqlite3.connect(BASE)
            c = conn.cursor()
            c.execute("INSERT INTO seances (nom,horodatage,categorie,mot,reaction,image_b64) VALUES (?,?,?,?,?,?)",
                     (nom, datetime.now().isoformat(), categorie, mot, message or "regarde écran", img_b64))
            conn.commit()
            conn.close()
    
    # Audio CORRIGÉ pour ElevenLabs v2
    audio_gen = client_eleven.text_to_speech.convert(
        text=reponse_claire,
        voice_id=ID_VOIX,
        model_id="eleven_turbo_v2_5",
        output_format="mp3_44100_128"
    )
    
    # Sauvegarde pour Gradio
    chemin_audio = "temp_audio.mp3"
    with open(chemin_audio, "wb") as f:
        for chunk in audio_gen:
            f.write(chunk)
    
    return reponse_claire, chemin_audio, images if images else None

# [Le reste du code reste identique : generer_rapport_semaine, interface Gradio, etc.]
# ... (copie le reste de ton fichier original ici pour la partie rapport et Gradio)

# Exemple pour la fin (ajoute si manquant)
with gr.Blocks(title="Claire Pro Max – Orthophoniste IA Autisme", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦋 Claire Pro Max\n**L'orthophoniste virtuelle que tous les enfants adorent**")
    
    with gr.Row():
        saisie_nom = gr.Textbox(label="Prénom de l'enfant", value="Léo", placeholder="Léo, Emma, Noah…")
        categorie = gr.Dropdown(
            choices=["famille", "animaux familiers", "nourriture", "actions", "objets & jouets", "émotions", "couleurs", "corps"],
            value="animaux familiers", label="Catégorie du jour"
        )
    
    chatbot = gr.Chatbot(height=500, avatar_images=["🧑‍⚕️", "👶"])
    saisie_msg = gr.Textbox(placeholder="L'enfant peut parler ici… ou rester silencieux, Claire continue quand même ❤️", label="Message / Observation (ex: regarde l'écran, pointe, vocalise, silence)")
    
    with gr.Row():
        audio = gr.Audio(label="Voix de Claire", autoplay=True, streaming=False, type="filepath")
        galerie = gr.Gallery(label="Photos générées aujourd'hui", height=400)
    
    btn_rapport = gr.Button("📄 Générer le rapport PDF de cette semaine", variant="primary")
    sortie_rapport = gr.File(label="Rapport téléchargeable")
    
    def mettre_a_jour_etat(nom):
        conn = sqlite3.connect(BASE)
        c = conn.cursor()
        c.execute("SELECT donnees FROM enfants WHERE nom=?", (nom,))
        ligne = c.fetchone()
        conn.close()
        if ligne:
            return json.dumps(json.loads(ligne[0]))
        else:
            defaut = {"nom": nom, "age_langage": "2 ans", "mots_maitrises": ["maman","papa"]}
            return json.dumps(defaut)
    
    def envoyer(message, historique, json_etat, cat):
        etat = json.loads(json_etat)
        reponse, chemin_audio, images = discuter_avec_claire(message, historique, etat, cat)
        historique.append(("👶 " + (message or "(regarde / silencieux)"), None))
        historique.append(("🧑‍⚕️ " + reponse, images[0] if images else None))
        return historique, chemin_audio, images or [], ""
    
    saisie_msg.submit(envoyer, [saisie_msg, chatbot, gr.State(value=""), categorie], [chatbot, audio, galerie, saisie_msg])
    
    btn_rapport.click(generer_rapport_semaine, saisie_nom, sortie_rapport)

demo.launch(share=True, server_port=7860)
