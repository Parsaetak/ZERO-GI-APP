# ZERO Execution Engine

The ZERO (Zero Error, Reliable Output) Execution Engine is a sophisticated, React-based web application that serves as an interactive chat interface for a high-assurance AI system. It's built on a metacognitive framework that enables an AI model (Google's Gemini) to critique and correct its own outputs, ensuring rigorous compliance with a predefined set of user-defined constraints.

The application is designed for scenarios where accuracy, auditable reasoning, and verifiable compliance are critical.

## Core Features

-   **Systematic Self-Correction:** Implements the ZERO framework's mandatory five-step workflow: Acknowledge, State Confidence (C4 Score), Draft, Await Critique, and Final Output.
-   **Persistent Session Management:** All conversations are saved to the browser's local storage. Users can create, load, rename, and delete sessions, ensuring no work is lost.
-   **Standing Constraints:** Users can define a persistent set of rules (e.g., "Always use a formal tone") that are automatically applied to every new task.
-   **URL Analysis:** The engine can analyze the content of external webpages by using its integrated Google Search tool to gather context and information.
-   **Self-Awareness Protocol:** The application can answer meta-questions about its own functionality and purpose by analyzing its own source code in a secure, redacted manner.
-   **Multi-modal Inputs:** Users can attach files (images, audio, video, PDFs) to their prompts for the AI to analyze.
-   **Rich Outputs:** AI responses are rendered with full markdown support, including professionally formatted code blocks with a one-click copy feature, and citations for web searches.
-   **Internationalization:** AI responses can be translated into dozens of languages on-demand.
-   **Conversation Export:** The entire chat log of a session can be exported as a `.txt` file for auditing or record-keeping.

## Setup and Running the Application

This application is designed to be run in a web environment where it can fetch its own files and has access to the Gemini API.

### Prerequisites

-   **API Key:** The application requires a valid Google Gemini API key to function. This key **must** be available as an environment variable named `process.env.API_KEY` in the execution context where the application is hosted. The application is hard-coded to read the key from this variable and will not work without it.

### Running the App

1.  Ensure all project files (`index.html`, `index.tsx`, `index.css`, `metadata.json`, `README.md`) are hosted on a web server.
2.  Set the `API_KEY` environment variable in your hosting environment.
3.  Open `index.html` in a web browser. The application will initialize, establish a connection with the Gemini API, and load your previous session or create a new one.

## Core ZERO Framework Concepts

-   **C4 Score (Confidence Score):** A predictive score from 0.00 to 1.00 where the AI estimates the probability of its first draft being fully compliant with all rules.
-   **Error Score:** A final score from 0.00 to 1.00 where the AI evaluates its final output against all rules. The goal is always `< 0.01`.
-   **ZEROSession Log:** A brief, auditable summary of the interaction steps, provided with every final output.

## Licensing

This application and its underlying framework are the sole and exclusive property of Parsa Tak and are protected by copyright and trade secret laws. Please review the full legal notice and license terms within the application by clicking the "Info" icon in the header.
