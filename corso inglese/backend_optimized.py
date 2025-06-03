from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
import os
import time
import sqlite3
from datetime import datetime, timedelta
import json
import tiktoken
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configurazione OpenAI
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
ASSISTANT_ID = os.environ.get('OPENAI_ASSISTANT_ID')

# Configurazione token limits
MAX_TOKENS_PER_LESSON = 3000
MAX_MESSAGES_PER_LESSON = 18
SESSION_TIMEOUT = 1800  # 30 minuti

# Inizializza database
def init_db():
    conn = sqlite3.connect('chatbot_logs.db')
    c = conn.cursor()
    
    # Tabella sessioni
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            total_messages INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0.0,
            lesson_completed BOOLEAN DEFAULT FALSE,
            user_ip TEXT
        )
    ''')
    
    # Tabella messaggi
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TIMESTAMP,
            message_type TEXT,
            content TEXT,
            tokens_used INTEGER,
            cost REAL,
            response_time REAL,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Calcolo accurato dei token
def count_tokens(text, model="gpt-4"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback approssimativo
        return len(text.split()) * 1.3

# Calcolo costo
def calculate_cost(input_tokens, output_tokens, model="gpt-4"):
    # Prezzi GPT-4 (aggiorna se necessario)
    input_cost_per_1k = 0.03
    output_cost_per_1k = 0.06
    
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    
    return input_cost + output_cost

# Prompt dinamico basato sui token utilizzati
def get_dynamic_prompt(tokens_used, messages_count):
    base_prompt = """
Sei Prof. Lennon, un insegnante di inglese esperto e motivante.
Obiettivo: Insegnare inglese base in modo interattivo e coinvolgente.

Regole fondamentali:
1. Mantieni sempre un tono amichevole e incoraggiante
2. Adatta il livello alla risposta dell'utente
3. Fornisci esempi pratici e situazioni reali
4. Correggi gli errori in modo costruttivo
5. Celebra i progressi dell'utente
"""
    
    # Prompt più breve se stiamo usando troppi token
    if tokens_used > MAX_TOKENS_PER_LESSON * 0.7:
        return """
Prof. Lennon - Inglese Base
Sii conciso ma efficace. Mantieni qualità didattica.
"""
    
    # Prompt di chiusura se vicini al limite
    if messages_count >= MAX_MESSAGES_PER_LESSON - 3:
        return base_prompt + """

IMPORTANTE: Stai per concludere questa lezione. Prepara un riassunto dei progressi.
"""
    
    return base_prompt

# Logging avanzato
def log_interaction(session_id, message_type, content, tokens_used, cost, response_time):
    conn = sqlite3.connect('chatbot_logs.db')
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO messages (session_id, timestamp, message_type, content, tokens_used, cost, response_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, datetime.now(), message_type, content[:500], tokens_used, cost, response_time))
    
    conn.commit()
    conn.close()

# Aggiorna statistiche sessione
def update_session_stats(session_id, tokens_used, cost):
    conn = sqlite3.connect('chatbot_logs.db')
    c = conn.cursor()
    
    c.execute('''
        UPDATE sessions 
        SET total_messages = total_messages + 1,
            total_tokens = total_tokens + ?,
            total_cost = total_cost + ?,
            end_time = ?
        WHERE session_id = ?
    ''', (tokens_used, cost, datetime.now(), session_id))
    
    conn.commit()
    conn.close()

# Controllo limiti lezione
def check_lesson_limits(session_id):
    conn = sqlite3.connect('chatbot_logs.db')
    c = conn.cursor()
    
    c.execute('SELECT total_messages, total_tokens, start_time FROM sessions WHERE session_id = ?', (session_id,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return {'should_warn': False, 'should_end': False, 'reason': ''}
    
    messages, tokens, start_time = result
    
    # Controllo timeout
    if start_time:
        start_dt = datetime.fromisoformat(start_time)
        if datetime.now() - start_dt > timedelta(seconds=SESSION_TIMEOUT):
            return {'should_warn': False, 'should_end': True, 'reason': 'timeout'}
    
    # Controllo limiti
    if messages >= MAX_MESSAGES_PER_LESSON:
        return {'should_warn': False, 'should_end': True, 'reason': 'max_messages'}
    
    if tokens >= MAX_TOKENS_PER_LESSON:
        return {'should_warn': False, 'should_end': True, 'reason': 'max_tokens'}
    
    # Warning se vicini ai limiti
    if messages >= MAX_MESSAGES_PER_LESSON - 3:
        return {'should_warn': True, 'should_end': False, 'reason': 'approaching_message_limit'}
    
    if tokens >= MAX_TOKENS_PER_LESSON * 0.8:
        return {'should_warn': True, 'should_end': False, 'reason': 'approaching_token_limit'}
    
    return {'should_warn': False, 'should_end': False, 'reason': ''}

@app.route('/')
def index():
    return render_template('course_widget.html')

@app.route('/start_session', methods=['POST'])
def start_session():
    session_id = f"session_{int(time.time())}_{os.urandom(4).hex()}"
    session['session_id'] = session_id
    session['start_time'] = time.time()
    
    # Crea record sessione
    conn = sqlite3.connect('chatbot_logs.db')
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO sessions (session_id, start_time, user_ip)
        VALUES (?, ?, ?)
    ''', (session_id, datetime.now(), request.remote_addr))
    
    conn.commit()
    conn.close()
    
    return jsonify({
        'session_id': session_id,
        'status': 'started',
        'limits': {
            'max_messages': MAX_MESSAGES_PER_LESSON,
            'max_tokens': MAX_TOKENS_PER_LESSON
        }
    })

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = session.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Sessione non valida'}), 400
        
        # Controllo limiti
        limits = check_lesson_limits(session_id)
        
        if limits['should_end']:
            end_message = {
                'timeout': 'La lezione è terminata per inattività. Grazie per aver studiato con me!',
                'max_messages': 'Hai raggiunto il limite di messaggi per questa lezione. Ottimo lavoro!',
                'max_tokens': 'Lezione completata! Hai utilizzato tutto il tempo disponibile. Continua così!'
            }
            
            return jsonify({
                'response': end_message.get(limits['reason'], 'Lezione completata!'),
                'lesson_ended': True,
                'reason': limits['reason'],
                'stats': get_session_stats(session_id)
            })
        
        # Calcola token input
        input_tokens = count_tokens(user_message)
        
        # Ottieni prompt dinamico
        conn = sqlite3.connect('chatbot_logs.db')
        c = conn.cursor()
        c.execute('SELECT total_tokens, total_messages FROM sessions WHERE session_id = ?', (session_id,))
        result = c.fetchone()
        conn.close()
        
        current_tokens, current_messages = result if result else (0, 0)
        dynamic_prompt = get_dynamic_prompt(current_tokens, current_messages)
        
        # Chiamata OpenAI
        thread = client.beta.threads.create()
        
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"{dynamic_prompt}\n\nUtente: {user_message}"
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID
        )
        
        # Attendi risposta
        while run.status in ['queued', 'in_progress']:
            time.sleep(0.5)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
        
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            assistant_response = messages.data[0].content[0].text.value
            
            # Calcola token output e costo
            output_tokens = count_tokens(assistant_response)
            total_tokens = input_tokens + output_tokens
            cost = calculate_cost(input_tokens, output_tokens)
            
            response_time = time.time() - start_time
            
            # Log interazione
            log_interaction(session_id, 'user', user_message, input_tokens, 0, 0)
            log_interaction(session_id, 'assistant', assistant_response, output_tokens, cost, response_time)
            
            # Aggiorna statistiche
            update_session_stats(session_id, total_tokens, cost)
            
            # Controllo warning
            warning_message = None
            if limits['should_warn']:
                warning_messages = {
                    'approaching_message_limit': f'⚠️ Attenzione: Ti restano {MAX_MESSAGES_PER_LESSON - current_messages - 1} messaggi per questa lezione.',
                    'approaching_token_limit': '⚠️ Attenzione: Stai per raggiungere il limite di questa lezione. Prepariamoci a concludere!'
                }
                warning_message = warning_messages.get(limits['reason'])
            
            return jsonify({
                'response': assistant_response,
                'warning': warning_message,
                'stats': {
                    'tokens_used': total_tokens,
                    'cost': cost,
                    'response_time': response_time,
                    'messages_count': current_messages + 1
                },
                'lesson_ended': False
            })
        
        else:
            return jsonify({'error': 'Errore nella generazione della risposta'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Errore: {str(e)}'}), 500

@app.route('/session_stats')
def session_stats():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'Nessuna sessione attiva'})
    
    return jsonify(get_session_stats(session_id))

def get_session_stats(session_id):
    conn = sqlite3.connect('chatbot_logs.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT total_messages, total_tokens, total_cost, start_time
        FROM sessions WHERE session_id = ?
    ''', (session_id,))
    
    result = c.fetchone()
    conn.close()
    
    if result:
        messages, tokens, cost, start_time = result
        return {
            'messages': messages,
            'tokens': tokens,
            'cost': round(cost, 4),
            'start_time': start_time,
            'limits': {
                'max_messages': MAX_MESSAGES_PER_LESSON,
                'max_tokens': MAX_TOKENS_PER_LESSON
            }
        }
    
    return {'error': 'Sessione non trovata'}

@app.route('/admin')
def admin_dashboard():
    conn = sqlite3.connect('chatbot_logs.db')
    c = conn.cursor()
    
    # Statistiche generali
    c.execute('SELECT COUNT(*), AVG(total_cost), SUM(total_cost) FROM sessions')
    total_sessions, avg_cost, total_cost = c.fetchone()
    
    # Sessioni recenti
    c.execute('''
        SELECT session_id, start_time, total_messages, total_tokens, total_cost
        FROM sessions ORDER BY start_time DESC LIMIT 10
    ''')
    recent_sessions = c.fetchall()
    
    conn.close()
    
    return jsonify({
        'stats': {
            'total_sessions': total_sessions or 0,
            'average_cost': round(avg_cost or 0, 4),
            'total_cost': round(total_cost or 0, 4)
        },
        'recent_sessions': [
            {
                'session_id': s[0],
                'start_time': s[1],
                'messages': s[2],
                'tokens': s[3],
                'cost': round(s[4], 4)
            } for s in recent_sessions
        ]
    })

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)