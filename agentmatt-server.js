// ─────────────────────────────────────────────────────────────────────────────
// AgentMatt Voice Check-In Server
// ─────────────────────────────────────────────────────────────────────────────
// Setup:
//   1. npm install (using agentmatt-package.json)
//   2. Copy .env.example → .agentmatt.env and fill in credentials
//   3. node agentmatt-server.js
//   4. Expose publicly: ngrok http 3001  →  set PUBLIC_URL in Settings UI
// ─────────────────────────────────────────────────────────────────────────────

require('dotenv').config({ path: '.agentmatt.env' });

const express  = require('express');
const cors     = require('cors');
const twilio   = require('twilio');
const Anthropic = require('@anthropic-ai/sdk');
const axios    = require('axios');
const crypto   = require('crypto');

const app  = express();
const PORT = process.env.PORT || 3001;

app.use(cors({ origin: '*' }));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.get('/', (_req, res) => res.sendFile(__dirname + '/agentmatt.html'));
app.use(express.static('.'));

// ─── Runtime Config (hydrated from env, overridable via UI) ──────────────────

let cfg = {
  twilioAccountSid:  process.env.TWILIO_ACCOUNT_SID   || '',
  twilioAuthToken:   process.env.TWILIO_AUTH_TOKEN    || '',
  twilioPhoneNumber: process.env.TWILIO_PHONE_NUMBER  || '',
  elevenLabsApiKey:  process.env.ELEVENLABS_API_KEY   || '',
  elevenLabsVoiceId: process.env.ELEVENLABS_VOICE_ID  || '',
  anthropicApiKey:   process.env.ANTHROPIC_API_KEY    || '',
  publicUrl:         process.env.PUBLIC_URL           || process.env.RENDER_EXTERNAL_URL || `http://localhost:${PORT}`,
  providerName:      process.env.PROVIDER_NAME        || 'Matt',
};

// ─── In-memory stores ────────────────────────────────────────────────────────

/** @type {Map<string, CallSession>} */
const activeCalls  = new Map();
/** @type {CallSession[]} */
const callHistory  = [];
/** @type {Map<string, {buffer: Buffer, created: number}>} */
const audioCache   = new Map();

// ─── Helpers ──────────────────────────────────────────────────────────────────

function twilioClient() {
  return twilio(cfg.twilioAccountSid, cfg.twilioAuthToken);
}

/**
 * Call ElevenLabs TTS, cache the audio, return a short-lived audioId.
 * Returns null if ElevenLabs is not configured.
 */
async function synthesize(text) {
  if (!cfg.elevenLabsApiKey || !cfg.elevenLabsVoiceId) return null;

  const res = await axios.post(
    `https://api.elevenlabs.io/v1/text-to-speech/${cfg.elevenLabsVoiceId}/stream`,
    {
      text,
      model_id: 'eleven_turbo_v2',
      voice_settings: {
        stability:        0.60,
        similarity_boost: 0.80,
        style:            0.00,
        use_speaker_boost: true,
      },
    },
    {
      headers: {
        'xi-api-key':   cfg.elevenLabsApiKey,
        'Content-Type': 'application/json',
        'Accept':       'audio/mpeg',
      },
      responseType: 'arraybuffer',
    }
  );

  const id = crypto.randomUUID();
  audioCache.set(id, { buffer: Buffer.from(res.data), created: Date.now() });
  setTimeout(() => audioCache.delete(id), 12 * 60 * 1000); // 12-min TTL
  return id;
}

/**
 * Build a TwiML response that plays (or says) text, then optionally gathers speech.
 */
function buildTwiML(audioId, sayText, gatherAction) {
  const VR = twilio.twiml.VoiceResponse;
  const r  = new VR();

  if (audioId) {
    r.play(`${cfg.publicUrl}/audio/${audioId}`);
  } else {
    // Fallback to Twilio neural TTS
    r.say({ voice: 'Polly.Matthew-Neural' }, sayText);
  }

  if (gatherAction) {
    const g = r.gather({
      input:             'speech',
      action:            gatherAction,
      method:            'POST',
      speechTimeout:     'auto',
      speechModel:       'experimental_conversations',
      language:          'en-US',
      timeout:           12,
    });
    g.pause({ length: 1 });
    // If patient doesn't speak, redirect back so we can handle the silence
    r.redirect({ method: 'POST' }, `${gatherAction}?silent=true`);
  } else {
    r.hangup();
  }

  return r.toString();
}

/**
 * Run the conversation through Claude.
 * Returns { text: string, shouldHangup: boolean }
 */
async function runClaude(session, patientUtterance) {
  const anthropic = new Anthropic({ apiKey: cfg.anthropicApiKey });

  const questionList = session.questions
    .map((q, i) => `  ${i + 1}. ${q}`)
    .join('\n');

  const nextQuestion =
    session.currentQuestionIndex < session.questions.length
      ? `"${session.questions[session.currentQuestionIndex]}"`
      : '(All questions asked)';

  const system = `You are ${cfg.providerName}, a caring healthcare provider making a brief phone check-in call.

PATIENT: ${session.patientName}
RISK LEVEL: ${session.riskLevel || 'High'}
NOTES: ${session.notes || 'None'}

CHECK-IN QUESTIONS (ask in order):
${questionList}

CURRENT STATE:
- Questions completed: ${session.currentQuestionIndex} / ${session.questions.length}
- Next question: ${nextQuestion}

INSTRUCTIONS:
- This is a live phone call — keep every response under 35 words.
- Speak naturally as yourself calling a patient you know.
- After the patient responds, briefly acknowledge it, then ask the next question.
- If the patient mentions a serious symptom or emergency, say: "That sounds urgent — please call 911 right away, and I'll have my office follow up with you immediately."
- Once all questions are answered AND you have given a closing statement, end with the exact token: [END_CALL]
- Do NOT include [END_CALL] unless you have already said a warm goodbye.

OUTPUT ONLY what you would say aloud on the phone. No meta-commentary.`;

  // Seed conversation on first turn
  const history = [...session.conversationHistory];
  if (history.length === 0) {
    history.push({
      role: 'user',
      content: '(The call just connected and the patient answered. Begin the check-in.)',
    });
  } else if (patientUtterance) {
    history.push({ role: 'user', content: patientUtterance });
  }

  const response = await anthropic.messages.create({
    model:      'claude-opus-4-6',
    max_tokens: 250,
    system,
    messages:   history,
  });

  let text = response.content[0].text.trim();
  const shouldHangup = text.includes('[END_CALL]');
  text = text.replace('[END_CALL]', '').trim();

  // Persist to session history
  if (patientUtterance) {
    session.conversationHistory.push({ role: 'user', content: patientUtterance });
  } else if (session.conversationHistory.length === 0) {
    session.conversationHistory.push({
      role: 'user',
      content: '(The call just connected and the patient answered. Begin the check-in.)',
    });
  }
  session.conversationHistory.push({ role: 'assistant', content: text });

  // Advance question pointer when patient has spoken
  if (patientUtterance && session.currentQuestionIndex < session.questions.length) {
    session.currentQuestionIndex++;
  }

  return { text, shouldHangup };
}

// ─── Config Routes ───────────────────────────────────────────────────────────

app.get('/api/config', (_req, res) => {
  const masked = (s) => (s ? '•'.repeat(8) : '');
  res.json({
    ...cfg,
    twilioAuthToken:  masked(cfg.twilioAuthToken),
    elevenLabsApiKey: masked(cfg.elevenLabsApiKey),
    anthropicApiKey:  masked(cfg.anthropicApiKey),
    isConfigured: !!(cfg.twilioAccountSid && cfg.twilioAuthToken && cfg.anthropicApiKey),
  });
});

app.post('/api/config', (req, res) => {
  const { twilioAuthToken, elevenLabsApiKey, anthropicApiKey, ...rest } = req.body;
  Object.assign(cfg, rest);
  if (twilioAuthToken  && !twilioAuthToken.includes('•'))  cfg.twilioAuthToken  = twilioAuthToken;
  if (elevenLabsApiKey && !elevenLabsApiKey.includes('•')) cfg.elevenLabsApiKey = elevenLabsApiKey;
  if (anthropicApiKey  && !anthropicApiKey.includes('•'))  cfg.anthropicApiKey  = anthropicApiKey;
  res.json({ success: true });
});

// ─── Call Management Routes ──────────────────────────────────────────────────

app.post('/api/calls/initiate', async (req, res) => {
  const { patientName, phone, questions, riskLevel, notes } = req.body;

  if (!cfg.twilioAccountSid || !cfg.twilioAuthToken || !cfg.twilioPhoneNumber) {
    return res.status(400).json({ error: 'Twilio credentials not configured' });
  }
  if (!cfg.anthropicApiKey) {
    return res.status(400).json({ error: 'Anthropic API key not configured' });
  }
  if (!questions || questions.length === 0) {
    return res.status(400).json({ error: 'At least one question is required' });
  }

  try {
    const client = twilioClient();
    const call   = await client.calls.create({
      to:                    phone,
      from:                  cfg.twilioPhoneNumber,
      url:                   `${cfg.publicUrl}/twilio/voice`,
      method:                'POST',
      statusCallback:        `${cfg.publicUrl}/twilio/status`,
      statusCallbackMethod:  'POST',
      statusCallbackEvent:   ['initiated', 'ringing', 'answered', 'completed'],
    });

    /** @type {CallSession} */
    const session = {
      callSid:              call.sid,
      patientName,
      phone,
      questions,
      riskLevel:            riskLevel || 'High',
      notes:                notes || '',
      currentQuestionIndex: 0,
      conversationHistory:  [],
      transcript:           [],
      state:                'INITIATED',
      status:               'initiated',
      startTime:            new Date().toISOString(),
      endTime:              null,
      outcome:              null,
    };

    activeCalls.set(call.sid, session);
    res.json({ callSid: call.sid, status: 'initiated' });
  } catch (err) {
    console.error('[initiate]', err.message);
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/calls', (_req, res) => {
  const active = Array.from(activeCalls.values());
  res.json([...active.reverse(), ...callHistory]);
});

app.get('/api/calls/:sid', (req, res) => {
  const session = activeCalls.get(req.params.sid)
    || callHistory.find(c => c.callSid === req.params.sid);
  if (!session) return res.status(404).json({ error: 'Not found' });
  res.json(session);
});

// ─── Twilio Webhook: Initial Call ────────────────────────────────────────────

app.post('/twilio/voice', async (req, res) => {
  const { CallSid } = req.body;
  const session = activeCalls.get(CallSid);

  if (!session) {
    const r = new twilio.twiml.VoiceResponse();
    r.say('Hello, this is a check-in call. We will call again shortly. Goodbye.');
    r.hangup();
    return res.type('text/xml').send(r.toString());
  }

  session.status = 'in-progress';
  session.state  = 'GREETING';

  try {
    const { text, shouldHangup } = await runClaude(session, null);
    session.transcript.push({ speaker: 'agent', text, ts: new Date().toISOString() });

    const audioId = await synthesize(text).catch(() => null);
    const action  = `${cfg.publicUrl}/twilio/gather/${CallSid}`;

    res.type('text/xml').send(buildTwiML(audioId, text, shouldHangup ? null : action));
  } catch (err) {
    console.error('[voice webhook]', err.message);
    const r = new twilio.twiml.VoiceResponse();
    r.say('Hello, this is your care team checking in. We will try again shortly.');
    r.hangup();
    res.type('text/xml').send(r.toString());
  }
});

// ─── Twilio Webhook: Speech Gather ──────────────────────────────────────────

app.post('/twilio/gather/:callSid', async (req, res) => {
  const { callSid }      = req.params;
  const { SpeechResult } = req.body;
  const silent           = req.query.silent === 'true';
  const session          = activeCalls.get(callSid);

  if (!session) {
    const r = new twilio.twiml.VoiceResponse();
    r.hangup();
    return res.type('text/xml').send(r.toString());
  }

  // Handle silence / no input
  if (silent && !SpeechResult) {
    const r    = new twilio.twiml.VoiceResponse();
    const text = "I'm sorry, I didn't catch that. Are you still there?";
    const id   = await synthesize(text).catch(() => null);
    if (id) r.play(`${cfg.publicUrl}/audio/${id}`);
    else    r.say({ voice: 'Polly.Matthew-Neural' }, text);
    const g = r.gather({
      input: 'speech',
      action: `${cfg.publicUrl}/twilio/gather/${callSid}`,
      method: 'POST',
      speechTimeout: 'auto',
      timeout: 12,
    });
    g.pause({ length: 1 });
    r.redirect({ method: 'POST' }, `${cfg.publicUrl}/twilio/gather/${callSid}?silent=true`);
    return res.type('text/xml').send(r.toString());
  }

  if (SpeechResult) {
    session.transcript.push({
      speaker: 'patient',
      text:    SpeechResult,
      ts:      new Date().toISOString(),
    });
  }

  session.state = 'PROCESSING';

  try {
    const { text, shouldHangup } = await runClaude(session, SpeechResult || '(no response)');
    session.transcript.push({ speaker: 'agent', text, ts: new Date().toISOString() });

    const audioId = await synthesize(text).catch(() => null);
    const action  = `${cfg.publicUrl}/twilio/gather/${callSid}`;

    if (shouldHangup) {
      session.state   = 'COMPLETED';
      session.status  = 'completed';
      session.endTime = new Date().toISOString();
      session.outcome = 'completed';
      callHistory.unshift({ ...session });
      activeCalls.delete(callSid);
      return res.type('text/xml').send(buildTwiML(audioId, text, null));
    }

    session.state = 'IN_CONVERSATION';
    res.type('text/xml').send(buildTwiML(audioId, text, action));
  } catch (err) {
    console.error('[gather]', err.message);
    const r    = new twilio.twiml.VoiceResponse();
    const bye  = 'Thank you for speaking with me today. Take care, and call us if you need anything.';
    const id   = await synthesize(bye).catch(() => null);
    if (id) r.play(`${cfg.publicUrl}/audio/${id}`);
    else    r.say({ voice: 'Polly.Matthew-Neural' }, bye);
    r.hangup();
    res.type('text/xml').send(r.toString());
  }
});

// ─── Twilio Webhook: Call Status ─────────────────────────────────────────────

app.post('/twilio/status', (req, res) => {
  const { CallSid, CallStatus } = req.body;
  const session = activeCalls.get(CallSid);

  if (session) {
    session.status = CallStatus;
    const terminal = ['completed', 'failed', 'no-answer', 'busy', 'canceled'];
    if (terminal.includes(CallStatus)) {
      session.endTime = new Date().toISOString();
      session.outcome = CallStatus;
      callHistory.unshift({ ...session });
      activeCalls.delete(CallSid);
    }
  }

  res.sendStatus(200);
});

// ─── Audio Serving ────────────────────────────────────────────────────────────

app.get('/audio/:id', (req, res) => {
  const entry = audioCache.get(req.params.id);
  if (!entry) return res.status(404).send('Audio expired or not found');
  res.set('Content-Type', 'audio/mpeg');
  res.set('Cache-Control', 'no-store');
  res.send(entry.buffer);
});

// ─── ElevenLabs Voice Routes ──────────────────────────────────────────────────

app.post('/api/voice/preview', async (req, res) => {
  const text = req.body.text || `Hi there, this is ${cfg.providerName}. Just calling to check in and see how you're doing today.`;
  try {
    const id = await synthesize(text);
    if (!id) return res.status(400).json({ error: 'ElevenLabs API key or Voice ID not configured' });
    res.json({ audioUrl: `${cfg.publicUrl}/audio/${id}` });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/voice/voices', async (_req, res) => {
  if (!cfg.elevenLabsApiKey) {
    return res.status(400).json({ error: 'ElevenLabs API key not configured' });
  }
  try {
    const r = await axios.get('https://api.elevenlabs.io/v1/voices', {
      headers: { 'xi-api-key': cfg.elevenLabsApiKey },
    });
    res.json(
      r.data.voices.map(v => ({
        id:       v.voice_id,
        name:     v.name,
        category: v.category || 'custom',
        preview:  v.preview_url,
      }))
    );
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── Health Check ─────────────────────────────────────────────────────────────

app.get('/api/health', (_req, res) => {
  res.json({
    status:      'ok',
    activesCalls: activeCalls.size,
    historyCount: callHistory.length,
    configured: {
      twilio:      !!(cfg.twilioAccountSid && cfg.twilioAuthToken && cfg.twilioPhoneNumber),
      elevenLabs:  !!(cfg.elevenLabsApiKey && cfg.elevenLabsVoiceId),
      anthropic:   !!cfg.anthropicApiKey,
    },
  });
});

// ─── Start ────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log('\n╔══════════════════════════════════════════╗');
  console.log('║   AgentMatt Voice Check-In Server  🎙️    ║');
  console.log('╚══════════════════════════════════════════╝');
  console.log(`\n  Local:  http://localhost:${PORT}`);
  console.log(`  UI:     open agentmatt.html in your browser\n`);
  console.log('  ⚠️  Twilio webhooks need a public URL.');
  console.log(`  Run:  ngrok http ${PORT}`);
  console.log('  Then paste the https URL into Settings → Public Server URL\n');
});
