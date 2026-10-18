# Lllra, the Little local literal research agent
Lllra, the little local literal research agent Python script, requires Ollama and a pre-pulled model.

## Author
Chris Pitchford admits to creating this mashup from multiple different sources and also using it!

## AI Use Disclosure
One of the tool functions was written by AI. When integrating the web scraping script into a tool function, I asked AI to review my work, since I hadn't had any breakfast yet.

## Documentation

### Purpose
The reasons to create this mini-agent harness were manifold.

1. Security

    1. No file reading
    1. No writing of files except by explicit instruction
    1. No web access except through APIs
2. Focused on research
3. Part of a larger system of editors for compiling results into a RAG system

### Use

1. Install dependencies
2. Run `python local_agent.py`
