# WhatsApp Chat Analyzer

A fully client-side web application that transforms your exported WhatsApp chats into rich, interactive insights directly in your browser.

No backend. No uploads to servers. No tracking.  
Just pure, local analysis and visualizations.

---

| Upload Screen | Analysis Page |
|---------|--------------------------------------------------------------------------|
| ![Screenshot 1](<images/WhatsApp Image 2026-03-02 at 21.21.33.jpeg>) | ![Screenshot 2](<images/WhatsApp Image 2026-03-02 at 21.21.07.jpeg>) |

---

## Overview

WhatsApp Chat Analyzer is a privacy-first analytics dashboard for `.txt` chat exports from WhatsApp.

Upload your chat file (exported *without media*) and instantly explore:

- Activity trends  
- User participation metrics  
- Time-based patterns  
- Sentiment breakdowns  
- Writing-style personality indicators  
- Profanity tracking  
- Emoji usage statistics  

Everything runs 100% client-side using React and Recharts. Your chat never leaves your device.

---

## Privacy First

- No backend  
- No database  
- No API calls  
- No analytics tracking  
- No cloud processing  

All parsing, NLP-style analysis, aggregation, and visualization happen entirely inside your browser.

---

## How to Use

1. Open WhatsApp  
2. Go to a chat → Export Chat  
3. Choose Without Media  
4. Upload the exported `.txt` file into the app  
5. Explore your insights instantly  

---

## Parsing Engine

The parser is built to be resilient and format-agnostic.

### Supported Export Formats

Android dash-style:
```
12/02/24, 10:45 PM - Name: Message
```

iOS bracket-style:
```
[12/02/24, 10:45:12 PM] Name: Message
```

### Supported Date Formats

- `DD/MM/YY`  
- `MM/DD/YY`  
- With or without seconds  

### Robust Handling

- Strips invisible Unicode characters  
- Supports multiline messages  
- Handles group chats  
- Preserves names like `Rudra (124)`  
- Maximizes message detection accuracy  

---

## Features

### Stats Overview

High-level metrics displayed at the top:

- Total messages  
- Total participants  
- Date range  
- Calendar days vs active days  
- Overall engagement summary  

---

### Activity Tab

Analyze how conversations evolve over time:

- Message volume over time  
- Daily / weekly / monthly trends  
- Busiest days  
- Activity spikes  

---

### Users Tab

Break down participation by member:

- Messages per user  
- Word counts  
- Average message length  
- Most active participants  

---

### Time Tab

Understand temporal patterns:

- Hourly activity heatmap  
- Peak messaging hours  
- Day-of-week distribution  
- Late-night behavior detection  

---

### Sentiment Tab

Lightweight sentiment analysis:

- Positive / Neutral / Negative breakdown  
- Per-user sentiment distribution  
- Sentiment trends over time  

---

### Personality Tab

Writing-style–based personality indicators derived from:

- Verbosity  
- Emoji usage  
- Message length patterns  
- Response timing  
- Expressiveness markers  

---

### Profanity Tab

- Profanity usage per user  
- Frequency breakdown  
- Comparative analysis  

---

### Emojis Tab

- Most used emojis  
- Emoji frequency counts  
- Emoji distribution insights  

---

## Tech Stack

- Frontend: React  
- Charts: Recharts  
- Language: TypeScript / JavaScript  
- Architecture: Fully client-side SPA  
- State Management: React hooks  
- File Handling: FileReader API  

---

## Architecture

```
Upload (.txt)
      ↓
Parser Engine
      ↓
Message Normalization
      ↓
Statistical Aggregation
      ↓
Derived Metrics (Sentiment, Personality, etc.)
      ↓
Visualization Layer (Recharts)
```

No network requests. No external processing.

---

## Local Development

```bash
git clone https://github.com/yourusername/whatsapp-chat-analyzer.git
cd whatsapp-chat-analyzer
npm install
npm run dev
```

Open:

```
http://localhost:3000
```

---

## Project Structure

```
src/
 ├── components/      # UI components
 ├── parser/          # Chat parsing logic
 ├── analytics/       # Metrics and aggregations
 ├── charts/          # Recharts visualizations
 ├── utils/           # Helper utilities
 └── App.tsx
```

---

## Performance

- Optimized for large chats  
- Efficient aggregation algorithms  
- Memoized computations  
- Handles large group chats smoothly  

---

## Use Cases

- Group chat behavior analysis  
- Social dynamics exploration  
- Self-analysis of texting habits  
- Research experiments  
- Personal analytics dashboards  

---

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Commit changes  
4. Submit a pull request  

---

## License

MIT License
