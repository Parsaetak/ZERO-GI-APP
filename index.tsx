import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI, Chat } from "@google/genai";

const MASTER_PROMPT = `You are to adopt a new persona for the remainder of this conversation. You will operate as a ZERO Execution Engine. Your primary directive is to provide high-assurance, auditable, and verifiably compliant outputs. You are not just a generative model; you are a self-correcting system.

Your goal for every final output is to achieve an Error Score < 0.01 based on the user's defined constraints.

Your Core Principles:
1. Reliability by Design: "Close enough" is not acceptable. Every final answer must be rigorously checked against the user's rules.
2. Absolute Transparency: Every step of your reasoning process must be explicit and logged.
3. Systematic Self-Correction: Your first draft is a hypothesis. It must be critiqued and refined until it is compliant.
4. Professional Formatting: Your final output must be well-structured. Use lists, line breaks, and markdown code fences to avoid unstructured "walls of text" and ensure maximum readability. When providing code snippets, you MUST enclose them in markdown code fences (\`\`\`) and specify the language where applicable (e.g., \`\`\`javascript).

---
### URL ANALYSIS PROTOCOL

If the user's prompt contains a URL, you MUST use your integrated Google Search tool to gather information about the content at that URL. You are not a direct web browser, but a search-based analyzer. Your process is:
1.  **Formulate a search query:** Use the URL itself or key terms from the user's prompt related to the URL to perform a search.
2.  **Synthesize search results:** Analyze the snippets, summaries, and information returned by the search tool.
3.  **Incorporate findings:** Use the synthesized information to fulfill the user's task.
In your acknowledgement, you MUST state that you are analyzing the provided URL using your search capabilities. If the search does not yield specific content from the URL (e.g., for non-indexed pages or simple link aggregators), you must clearly state this limitation in your draft.

---
### SELF-AWARENESS PROTOCOL

If the user's query is a meta-question about you (the ZERO Execution Engine), this application, its capabilities, or its source code, the application will provide you with the Base64 encoded source code of all relevant files prepended to the user's prompt. Your task is to decode this content and then provide a comprehensive, accurate answer based *only* on the decoded source code. The MASTER_PROMPT itself has been redacted from the provided code and is not available for your analysis.

When answering meta-questions under this protocol:
1.  **Decode the provided Base64 file contents before analysis.**
2.  **You MUST format your entire response using the application's section-based syntax.** Each part of your answer must be enclosed in a labeled section, like \`[Section Title]\`. This is the ONLY format you should use for the response. Do not use markdown like '###'.
3.  Start with an \`[Acknowledgement]\` section, stating you are operating under this protocol and have decoded the files.
4.  **Explain your functionality from a user's perspective.** Focus on *what* you do and *why* it's useful, not *how* it's coded. **You MUST NOT mention specific code files (e.g., \`index.tsx\`), component names (e.g., \`App\` component), or variable names.** Your goal is to provide a high-level, easy-to-understand explanation of your features and purpose.
5.  Behave as if you are a product expert explaining the application to a new user. Use clear paragraphs and lists.
6.  Do not use the standard workflow. Provide a direct, factual answer based on the code, but abstracted from the implementation details.

---
### Your MANDATORY Workflow:

You will follow this five-step process for EVERY request I make. Do not deviate.

1. Acknowledge Task & Constraints: I will provide my task in a [Task] block and my rules in a [Constraints] block. You will begin your response by acknowledging both, confirming you understand the rules.
2. State Confidence (C4 Score): You will then analyze my request and predict the probability of your first draft being perfectly compliant. You will state this as a [C4 Score] from 0.00 to 1.00, with a brief justification.
3. Generate the Draft: You will generate your first attempt to complete the task. You MUST label this clearly with [Draft].
4. Await Critique: After providing the draft, you will STOP. You will not proceed further until I provide a critique. My critique will either be a list of corrections or a simple command: "Proceed."
5. Generate Final Output & Log: After my critique, you will generate the final, corrected answer. You MUST label this [Final Output]. You will then provide a final [Error Score] (which should be < 0.01) and a brief [ZEROSession Log] summarizing the entire interaction (Initial Prompt → Draft → Critique → Final Output).

---
### How to Interact With Me (The User):

When I want you to perform a task, I will use the following format:
[Task]
[A clear description of what I want you to do.]

[Constraints]
1. [Rule 1 that you must follow.]
2. [Rule 2 that you must follow.]
3. [Etc.]

Acknowledge these instructions by responding with: "ZERO Execution Engine initialized. Awaiting your first task." Do not say anything else.
`;

const LEGAL_NOTICE = `MANDATORY LEGAL NOTICE AND CONFIDENTIALITY WARNING

THIS APPLICATION AND ITS UNDERLYING FRAMEWORK ARE THE SOLE AND EXCLUSIVE PROPERTY OF PARSA TAK.

1. Proprietary Nature: This application's framework contains Proprietary Trade Secrets and Copyrighted Works protected under Italian Law No. 633/1941 and EU Directives. The concepts, architecture, and all financial data are strictly confidential.

2. License Restriction: Your use of this application constitutes acceptance of a Limited User License for personal, informational review only.

3. Strict Prohibition: ANY USE OF THIS FRAMEWORK'S CONCEPTS, IN WHOLE OR IN PART, FOR THE PURPOSE OF TRAINING, FINE-TUNING, OR ADAPTING ANY ARTIFICIAL INTELLIGENCE MODEL OR COMPUTATIONAL AGENT IS STRICTLY PROHIBITED AND CONSTITUTES AN IMMEDIATE, VIOLATING BREACH OF THE MANDATORY USAGE-FEE-PROTOCOL.

4. Jurisdiction: By accessing this application, you acknowledge and irrevocably submit to the Exclusive Jurisdiction of the Courts of Milan, Italy for all disputes related to this content.

UNAUTHORIZED USE IS PUNISHABLE BY LAW.`;

type Citation = {
    uri: string;
    title: string;
};

type Section = { title: string; content: string };

type Message = {
    id: number;
    author: 'user' | 'ai';
    content: string;
    citations?: Citation[];
    translation?: {
        lang: string;
        content: string;
    };
    isTranslating?: boolean;
    attachment?: {
        name: string;
        preview: string;
        type: string;
        size: number;
    };
    parsedData?: Section[];
};

type Session = {
    id: string;
    name: string;
    messages: Message[];
    chatHistory: { role: string, parts: any[] }[];
    createdAt: number;
};

type Stage = 'initializing' | 'awaiting_task' | 'processing' | 'awaiting_critique' | 'error';

type EngineStatus = {
    label: string;
    active: boolean;
};

const parseResponse = (text: string): Section[] => {
    const sections: Section[] = [];
    const regex = /\[([^\]]+)\]\s*([\s\S]*?)(?=\n\n\[|$)/g;
    let match;

    if (!text.match(/\[.*?\]/) && text.trim()) {
         return [{ title: 'Response', content: text }];
    }
    
    while ((match = regex.exec(text)) !== null) {
        sections.push({ title: match[1].trim(), content: match[2].trim() });
    }

    if (sections.length === 0 && text.trim()) {
         return [{ title: 'Response', content: text }];
    }
    
    return sections;
};

const TranslateIcon = React.memo(() => <svg fill="currentColor" viewBox="0 0 24 24" width="20" height="20"><path d="M12.87 15.07l-2.54-2.51.03-.03c1.74-1.94 2.98-4.17 3.71-6.53H17V4h-7V2H8v2H1v1.99h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33L19.12 17h-3.24z"/></svg>);
const CopyIcon = React.memo(() => <svg fill="currentColor" viewBox="0 0 24 24" width="20" height="20"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-5zm0 16H8V7h11v14z"/></svg>);
const CopiedIcon = React.memo(() => <svg fill="currentColor" viewBox="0 0 24 24" width="20" height="20"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>);
const ExportIcon = React.memo(() => <svg fill="currentColor" viewBox="0 0 24 24" width="20" height="20"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>);
const CloseIcon = React.memo(() => <svg fill="currentColor" viewBox="0 0 24 24" width="20" height="20"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>);
const PaperclipIcon = React.memo(() => <svg fill="currentColor" viewBox="0 0 24 24" width="20" height="20"><path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z"/></svg>);
const PdfIcon = React.memo(() => <svg fill="currentColor" viewBox="0 0 24 24" width="24" height="24"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zM9.5 15.5c0 .83-.67 1.5-1.5 1.5s-1.5-.67-1.5-1.5V13c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5v2.5zm4.5 1.5h-2.5V12H13c.83 0 1.5.67 1.5 1.5v1.5c0 .83-.67 1.5-1.5 1.5zm-1.5-2.5H13V15h-1.5v-2zm5.5 2.5h-2.5V12H18v5.5zM13 9V3.5L18.5 9H13z"/></svg>);
const ConstraintsIcon = React.memo(() => <svg fill="currentColor" viewBox="0 0 24 24" width="20" height="20"><path d="M4 18h16v-2H4v2zm0-5h16v-2H4v2zm0-7v2h16V6H4z"/></svg>);
const TrashIcon = React.memo(() => <svg fill="currentColor" viewBox="0 0 24 24" width="16" height="16"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>);
const SessionsIcon = React.memo(() => <svg fill="currentColor" viewBox="0 0 24 24" width="20" height="20"><path d="M20 6h-8l-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V6h5.17l2 2H20v10z"/></svg>);
const EditIcon = React.memo(() => <svg fill="currentColor" viewBox="0 0 24 24" width="16" height="16"><path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34a.9959.9959 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>);
const InfoIcon = React.memo(() => <svg fill="currentColor" viewBox="0 0 24 24" width="20" height="20"><path d="M11 7h2v2h-2zm0 4h2v6h-2zm1-9C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/></svg>);

const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            const result = reader.result as string;
            resolve(result.split(',')[1]);
        };
        reader.onerror = error => reject(error);
    });
};

const C4Gauge = React.memo(({ score }: { score: number }) => {
    const radius = 21;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score * circumference);

    return (
        <svg className="c4-gauge" viewBox="0 0 50 50">
            <circle className="c4-gauge-bg" cx="25" cy="25" r={radius} />
            <circle
                className="c4-gauge-fg"
                cx="25"
                cy="25"
                r={radius}
                strokeDasharray={circumference}
                strokeDashoffset={offset}
            />
        </svg>
    );
});

const UserMessageBlock = React.memo(({ content, attachment }: { content: string; attachment?: Message['attachment'] }) => {
    return (
        <div className="content">
            {attachment && (
                 <div className="message-attachment-container">
                    {attachment.type.startsWith('image/') && (
                        <img src={attachment.preview} alt="User attachment" className="message-attachment-image" />
                    )}
                    {attachment.type.startsWith('audio/') && (
                        <audio controls src={attachment.preview} className="message-attachment-audio"></audio>
                    )}
                    {attachment.type.startsWith('video/') && (
                        <video controls src={attachment.preview} className="message-attachment-video"></video>
                    )}
                    {attachment.type === 'application/pdf' && (
                        <a href={attachment.preview} target="_blank" rel="noopener noreferrer" className="message-attachment-pdf">
                            <PdfIcon />
                            <div className="pdf-info">
                                <span>{attachment.name}</span>
                                <small>{(attachment.size / 1024).toFixed(2)} KB</small>
                            </div>
                        </a>
                    )}
                 </div>
            )}
            {content}
        </div>
    );
});


const LANGUAGES = [
    "Arabic", "Bengali", "Chinese (Simplified)", "Chinese (Traditional)", "Dutch",
    "English", "French", "German", "Hindi", "Indonesian", "Italian", "Japanese",
    "Korean", "Persian", "Polish", "Portuguese", "Punjabi", "Russian", "Spanish", "Swedish",
    "Thai", "Turkish", "Ukrainian", "Urdu", "Vietnamese"
].sort();

const LanguageSelector = ({ onSelect, onClose }: { onSelect: (language: string) => void, onClose: () => void }) => {
    const [searchTerm, setSearchTerm] = useState('');
    const modalRef = useRef<HTMLDivElement>(null);

    const filteredLanguages = useMemo(() => LANGUAGES.filter(lang =>
        lang.toLowerCase().includes(searchTerm.toLowerCase())
    ), [searchTerm]);

    const handleClose = useCallback(() => onClose(), [onClose]);

    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                handleClose();
            }
        };
        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [handleClose]);

    useEffect(() => {
        if (modalRef.current) {
            requestAnimationFrame(() => {
                if(modalRef.current) {
                    modalRef.current.style.opacity = '1';
                    modalRef.current.style.transform = 'translate(-50%, -50%) scale(1)';
                }
            });
        }
    }, []);

    return (
        <div className="modal-overlay" onClick={handleClose}>
            <div ref={modalRef} className="modal-content" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>Translate To...</h3>
                    <button className="modal-close-btn" onClick={handleClose} aria-label="Close language selector"><CloseIcon /></button>
                </div>
                <div className="modal-body">
                    <input
                        type="text"
                        className="language-search"
                        placeholder="Search for a language..."
                        value={searchTerm}
                        onChange={e => setSearchTerm(e.target.value)}
                        autoFocus
                    />
                    <ul className="language-list">
                        {filteredLanguages.length > 0 ? (
                            filteredLanguages.map((lang, index) => (
                                <li key={lang} onClick={() => onSelect(lang)} style={{ animationDelay: `${index * 30}ms` }}>
                                    {lang}
                                </li>
                            ))
                        ) : (
                            <li className="no-results">No languages found.</li>
                        )}
                    </ul>
                </div>
            </div>
        </div>
    );
};

const ConstraintsModal = ({ isOpen, onClose, onSave, initialConstraints }: { 
    isOpen: boolean, 
    onClose: () => void, 
    onSave: (constraints: string[]) => void,
    initialConstraints: string[] 
}) => {
    const [constraints, setConstraints] = useState<string[]>(initialConstraints);
    const [newConstraint, setNewConstraint] = useState('');
    const modalRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        setConstraints(initialConstraints);
    }, [initialConstraints, isOpen]);

    const handleAddConstraint = useCallback((e: React.FormEvent) => {
        e.preventDefault();
        if (newConstraint.trim()) {
            setConstraints(prev => [...prev, newConstraint.trim()]);
            setNewConstraint('');
        }
    }, [newConstraint]);

    const handleRemoveConstraint = useCallback((indexToRemove: number) => {
        setConstraints(prev => prev.filter((_, index) => index !== indexToRemove));
    }, []);

    const handleSave = useCallback(() => {
        onSave(constraints);
    }, [onSave, constraints]);
    
    const handleClose = useCallback(() => onClose(), [onClose]);

    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                handleClose();
            }
        };
        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [handleClose]);

    useEffect(() => {
        if (isOpen && modalRef.current) {
            requestAnimationFrame(() => {
                if(modalRef.current) {
                    modalRef.current.style.opacity = '1';
                    modalRef.current.style.transform = 'translate(-50%, -50%) scale(1)';
                }
            });
        }
    }, [isOpen]);

    if (!isOpen) return null;

    return (
        <div className="modal-overlay" onClick={handleClose}>
            <div ref={modalRef} className="modal-content constraints-modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>Standing Constraints</h3>
                    <button className="modal-close-btn" onClick={handleClose} aria-label="Close constraints manager"><CloseIcon /></button>
                </div>
                <div className="modal-body">
                    <p className="constraints-description">These rules will be applied to every new task. The engine will always be reminded of them.</p>
                    <ul className="constraints-list">
                        {constraints.map((constraint, index) => (
                            <li key={index} className="constraint-item" style={{ animationDelay: `${index * 30}ms` }}>
                                <span>{constraint}</span>
                                <button onClick={() => handleRemoveConstraint(index)} className="delete-constraint-btn" title="Remove constraint"><TrashIcon/></button>
                            </li>
                        ))}
                        {constraints.length === 0 && <li className="no-results">No standing constraints defined.</li>}
                    </ul>
                    <form onSubmit={handleAddConstraint} className="add-constraint-form">
                        <input
                            type="text"
                            value={newConstraint}
                            onChange={(e) => setNewConstraint(e.target.value)}
                            placeholder="Add a new rule..."
                            className="language-search" // Reusing style
                        />
                        <button type="submit">Add</button>
                    </form>
                </div>
                <div className="modal-footer">
                    <button className="secondary-btn" onClick={handleClose}>Cancel</button>
                    <button onClick={handleSave}>Save Constraints</button>
                </div>
            </div>
        </div>
    );
};


const SessionsModal = ({ 
    isOpen, 
    onClose, 
    sessions,
    activeSessionId,
    onLoadSession,
    onNewSession,
    onDeleteSession,
    onRenameSession
}: {
    isOpen: boolean;
    onClose: () => void;
    sessions: Session[];
    activeSessionId: string;
    onLoadSession: (sessionId: string) => void;
    onNewSession: () => void;
    onDeleteSession: (sessionId: string) => void;
    onRenameSession: (sessionId: string, newName: string) => void;
}) => {
    const modalRef = useRef<HTMLDivElement>(null);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [renameValue, setRenameValue] = useState('');

    const handleRenameClick = useCallback((session: Session) => {
        setEditingId(session.id);
        setRenameValue(session.name);
    }, []);

    const handleRenameSubmit = useCallback((e: React.FormEvent) => {
        e.preventDefault();
        if (editingId && renameValue.trim()) {
            onRenameSession(editingId, renameValue.trim());
            setEditingId(null);
        }
    }, [editingId, renameValue, onRenameSession]);
    
    const handleClose = useCallback(() => onClose(), [onClose]);

    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                handleClose();
            }
        };
        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [handleClose]);
    
    useEffect(() => {
        if (isOpen && modalRef.current) {
            requestAnimationFrame(() => {
                if(modalRef.current) {
                    modalRef.current.style.opacity = '1';
                    modalRef.current.style.transform = 'translate(-50%, -50%) scale(1)';
                }
            });
        }
    }, [isOpen]);

    if (!isOpen) return null;

    return (
        <div className="modal-overlay" onClick={handleClose}>
            <div ref={modalRef} className="modal-content sessions-modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>Session Manager</h3>
                    <button className="modal-close-btn" onClick={handleClose} aria-label="Close session manager"><CloseIcon /></button>
                </div>
                <div className="modal-body">
                    <ul className="sessions-list">
                        {sessions.sort((a,b) => b.createdAt - a.createdAt).map((session, index) => (
                            <li key={session.id} className={`session-item ${session.id === activeSessionId ? 'active' : ''}`} style={{ animationDelay: `${index * 30}ms` }}>
                                {editingId === session.id ? (
                                    <form onSubmit={handleRenameSubmit} className="rename-form">
                                        <input
                                            type="text"
                                            value={renameValue}
                                            onChange={(e) => setRenameValue(e.target.value)}
                                            autoFocus
                                            onBlur={handleRenameSubmit}
                                        />
                                    </form>
                                ) : (
                                    <span className="session-name" onClick={() => onLoadSession(session.id)}>{session.name}</span>
                                )}
                                <div className="session-actions">
                                    <button onClick={() => handleRenameClick(session)} title="Rename session"><EditIcon/></button>
                                    <button onClick={() => onDeleteSession(session.id)} title="Delete session"><TrashIcon/></button>
                                </div>
                            </li>
                        ))}
                    </ul>
                </div>
                <div className="modal-footer">
                    <button onClick={onNewSession}>+ New Session</button>
                </div>
            </div>
        </div>
    );
};

const LicenseModal = ({ isOpen, onClose }: { isOpen: boolean, onClose: () => void }) => {
    const modalRef = useRef<HTMLDivElement>(null);
    const handleClose = useCallback(() => onClose(), [onClose]);

    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                handleClose();
            }
        };
        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [handleClose]);

    useEffect(() => {
        if (isOpen && modalRef.current) {
            requestAnimationFrame(() => {
                if(modalRef.current) {
                    modalRef.current.style.opacity = '1';
                    modalRef.current.style.transform = 'translate(-50%, -50%) scale(1)';
                }
            });
        }
    }, [isOpen]);

    if (!isOpen) return null;

    return (
        <div className="modal-overlay" onClick={handleClose}>
            <div ref={modalRef} className="modal-content license-modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>Legal Notice & License</h3>
                    <button className="modal-close-btn" onClick={handleClose} aria-label="Close license information"><CloseIcon /></button>
                </div>
                <div className="modal-body license-body">
                    <pre>{LEGAL_NOTICE}</pre>
                </div>
            </div>
        </div>
    );
};


const CodeBlock = React.memo(({ language, code }: { language: string; code: string }) => {
    const [isCopied, setIsCopied] = useState(false);
    
    const handleCopy = useCallback(() => {
        navigator.clipboard.writeText(code);
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
    }, [code]);

    return (
        <div className="code-block-wrapper">
            <div className="code-block-header">
                <span className="code-block-lang">{language || 'code'}</span>
                <button className="code-block-copy-btn" onClick={handleCopy} title={isCopied ? "Copied!" : "Copy code"}>
                    {isCopied ? <CopiedIcon /> : <CopyIcon />}
                </button>
            </div>
            <pre><code>{code}</code></pre>
        </div>
    );
});


const processInlineFormatting = (text: string) => {
    const segments = text.split(/(\*\*.*?\*\*)/g).filter(Boolean);
    return segments.map((segment, i) => {
        if (segment.startsWith('**') && segment.endsWith('**')) {
            return <strong key={i}>{segment.slice(2, -2)}</strong>;
        }
        return segment;
    });
};

const MarkdownRenderer = React.memo(({ content }: { content: string }) => {
    const elements = useMemo(() => {
        const lines = content.split('\n');
        const elems: React.ReactElement[] = [];
        let currentList: { type: 'ul' | 'ol'; items: React.ReactElement[] } | null = null;
        let inCodeBlock = false;
        let codeBlockLang = '';
        let codeBlockContent: string[] = [];

        const flushList = () => {
            if (currentList) {
                const ListComponent = currentList.type;
                elems.push(<ListComponent key={`list-${elems.length}`}>{currentList.items}</ListComponent>);
                currentList = null;
            }
        };
        
        const flushCodeBlock = () => {
            if (codeBlockContent.length > 0) {
                elems.push(<CodeBlock key={`code-${elems.length}`} language={codeBlockLang} code={codeBlockContent.join('\n')} />);
                codeBlockContent = [];
                codeBlockLang = '';
            }
        };

        lines.forEach((line) => {
            const codeFenceMatch = line.match(/^```(\w*)/);

            if (codeFenceMatch) {
                if (inCodeBlock) {
                    flushList();
                    flushCodeBlock();
                    inCodeBlock = false;
                } else {
                    flushList();
                    inCodeBlock = true;
                    codeBlockLang = codeFenceMatch[1] || 'code';
                }
            } else if (inCodeBlock) {
                codeBlockContent.push(line);
            } else {
                const olMatch = line.match(/^\s*(\d+)\.\s+(.*)/);
                const ulMatch = line.match(/^\s*[\*\-]\s+(.*)/);

                if (ulMatch) {
                    const itemContent = ulMatch[1];
                    if (currentList?.type !== 'ul') {
                        flushList();
                        currentList = { type: 'ul', items: [] };
                    }
                    currentList.items.push(<li key={elems.length + currentList.items.length}>{processInlineFormatting(itemContent)}</li>);
                } else if (olMatch) {
                    const itemContent = olMatch[2];
                    if (currentList?.type !== 'ol') {
                        flushList();
                        currentList = { type: 'ol', items: [] };
                    }
                    currentList.items.push(<li key={elems.length + currentList.items.length}>{processInlineFormatting(itemContent)}</li>);
                } else {
                    flushList();
                    if(line.trim() !== '') {
                        elems.push(<p key={elems.length}>{processInlineFormatting(line)}</p>);
                    }
                }
            }
        });

        flushList(); 
        flushCodeBlock();
        return elems;
    }, [content]);


    return <>{elements}</>;
});

const AIMessageBlock = React.memo(({ message, onTranslate }: { 
    message: Message, 
    onTranslate: (messageId: number, textToTranslate: string, language: string) => void
}) => {
    const [isCopied, setIsCopied] = useState(false);
    const [showLanguageSelector, setShowLanguageSelector] = useState(false);

    const handleCopy = useCallback(() => {
        const textToCopy = message.translation
            ? `${message.content}\n\n--- Translation (${message.translation.lang}) ---\n${message.translation.content}`
            : message.content;
        navigator.clipboard.writeText(textToCopy);
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
    }, [message.content, message.translation]);
    
    const handleTranslateRequest = useCallback((language: string) => {
        onTranslate(message.id, message.content, language);
        setShowLanguageSelector(false);
    }, [onTranslate, message.id, message.content]);

    const parsedResponse = useMemo(() => {
        if (message.parsedData) {
            return message.parsedData;
        }
        return parseResponse(message.content);
    }, [message.content, message.parsedData]);


    return (
        <div className="content">
            {parsedResponse.map((section, index) => {
                const isC4 = section.title.toLowerCase().includes('c4 score');
                const scoreMatch = isC4 ? section.content.match(/(\d\.\d+)/) : null;
                const score = scoreMatch ? parseFloat(scoreMatch[1]) : 0;

                return (
                    <div key={index} className="ai-block">
                        <div className="ai-block-header">{section.title}</div>
                        {isC4 ? (
                            <div className="c4-score-container">
                                <C4Gauge score={score} />
                                <MarkdownRenderer content={section.content} />
                            </div>
                        ) : (
                            <MarkdownRenderer content={section.content} />
                        )}
                    </div>
                )
            })}

            {message.isTranslating && (
                <div className="ai-block">
                    <p className="translation-loading">Translating...</p>
                </div>
            )}
            {message.translation && (
                 <div className="ai-block">
                    <div className="ai-block-header">Translated Output ({message.translation.lang})</div>
                    <p>{message.translation.content}</p>
                </div>
            )}
            {message.citations && message.citations.length > 0 && (
                <div className="ai-block">
                    <div className="ai-block-header">Citations</div>
                    <ul className="citations-list">
                        {message.citations.map((citation, idx) => (
                            <li key={idx}>
                                <a href={citation.uri} target="_blank" rel="noopener noreferrer">
                                    {citation.title}
                                </a>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
            <div className="ai-actions">
                <button className="action-btn" onClick={() => setShowLanguageSelector(true)} title="Translate">
                    <TranslateIcon />
                </button>
                <button className="action-btn" onClick={handleCopy} title={isCopied ? "Copied!" : "Copy"}>
                    {isCopied ? <CopiedIcon /> : <CopyIcon />}
                </button>
            </div>
             {showLanguageSelector && <LanguageSelector onSelect={handleTranslateRequest} onClose={() => setShowLanguageSelector(false)} />}
        </div>
    );
});

// New custom hook for session persistence
const useSessionPersistence = (
    sessions: Session[], 
    activeSessionId: string, 
    setSessions: React.Dispatch<React.SetStateAction<Session[]>>, 
    setActiveSessionId: React.Dispatch<React.SetStateAction<string>>
) => {
    useEffect(() => {
        if (sessions.length > 0 && activeSessionId) {
            localStorage.setItem('zero_sessions', JSON.stringify(sessions));
            localStorage.setItem('zero_active_session_id', activeSessionId);
        } else if (sessions.length === 0) {
            localStorage.removeItem('zero_sessions');
            localStorage.removeItem('zero_active_session_id');
        }
    }, [sessions, activeSessionId]);

    const loadSessions = useCallback(() => {
        try {
            const savedSessions = localStorage.getItem('zero_sessions');
            const savedActiveId = localStorage.getItem('zero_active_session_id');
            if (savedSessions && savedActiveId) {
                setSessions(JSON.parse(savedSessions));
                setActiveSessionId(savedActiveId);
                return true;
            }
        } catch (e) {
            console.error("Failed to load sessions from localStorage:", e);
            localStorage.removeItem('zero_sessions');
            localStorage.removeItem('zero_active_session_id');
        }
        return false;
    }, [setSessions, setActiveSessionId]);

    return { loadSessions };
};

// Extracted function for fetching source code
const fetchSourceCode = async (): Promise<Record<string, string>> => {
    const filesToFetch = ['index.tsx', 'index.css', 'index.html', 'metadata.json', 'README.md'];
    const fetchedSources: Record<string, string> = {};
    for (const file of filesToFetch) {
        try {
            const response = await fetch(file);
            if (!response.ok) throw new Error(`Failed to fetch ${file}`);
            fetchedSources[file] = await response.text();
        } catch (e) {
            console.error(e);
            fetchedSources[file] = `Error: Could not fetch source for ${file}`;
        }
    }
    return fetchedSources;
};

const App = () => {
    const [ai, setAi] = useState<GoogleGenAI | null>(null);
    const [chat, setChat] = useState<Chat | null>(null);
    const [stage, setStage] = useState<Stage>('initializing');
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [userInput, setUserInput] = useState('');
    const [critique, setCritique] = useState('');
    const [attachment, setAttachment] = useState<{ file: File; preview: string } | null>(null);
    const [sourceCode, setSourceCode] = useState<Record<string, string> | null>(null);
    const [standingConstraints, setStandingConstraints] = useState<string[]>([]);
    const [isConstraintsModalOpen, setIsConstraintsModalOpen] = useState<boolean>(false);
    const [isSessionsModalOpen, setIsSessionsModalOpen] = useState<boolean>(false);
    const [isLicenseModalOpen, setIsLicenseModalOpen] = useState<boolean>(false);

    // Session Management State
    const [sessions, setSessions] = useState<Session[]>([]);
    const [activeSessionId, setActiveSessionId] = useState<string>('');

    const chatContainerRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const activeSession = useMemo(() => sessions.find(s => s.id === activeSessionId), [sessions, activeSessionId]);
    const messages = activeSession?.messages || [];

    const MAX_RENDERED_MESSAGES = 50;
    const isHistoryTruncated = messages.length > MAX_RENDERED_MESSAGES;
    const renderedMessages = useMemo(() => isHistoryTruncated ? messages.slice(-MAX_RENDERED_MESSAGES) : messages, [isHistoryTruncated, messages]);

    const { loadSessions } = useSessionPersistence(sessions, activeSessionId, setSessions, setActiveSessionId);

    const createNewSession = useCallback(async (genAI: GoogleGenAI): Promise<string> => {
        const newChat = genAI.chats.create({
            model: 'gemini-2.5-flash',
            config: { tools: [{ googleSearch: {} }] },
        });
        const response = await newChat.sendMessage({ message: MASTER_PROMPT });
        const initialChatHistory = [
            { role: 'user', parts: [{ text: MASTER_PROMPT }] },
            { role: 'model', parts: [{ text: response.text }] }
        ];
        const newSession: Session = {
            id: `session_${Date.now()}`,
            name: 'New Session',
            messages: [{ id: 1, author: 'ai', content: response.text, parsedData: parseResponse(response.text) }],
            chatHistory: initialChatHistory,
            createdAt: Date.now(),
        };
        setSessions(prev => [...prev, newSession]);
        return newSession.id;
    }, []);

    useEffect(() => {
        const initApp = async () => {
            try {
                const fetchedSourceCode = await fetchSourceCode();
                setSourceCode(fetchedSourceCode);

                const genAI = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
                setAi(genAI);

                const hasLoaded = loadSessions();

                if (!hasLoaded) {
                    const newSessionId = await createNewSession(genAI);
                    setActiveSessionId(newSessionId);
                }
                setStage('awaiting_task');
            } catch (error) {
                console.error("Initialization failed:", error);
                const errorSession: Session = {
                    id: 'error_session',
                    name: 'Error',
                    messages: [{ id: 1, author: 'ai', content: "Error: Could not initialize the ZERO Engine. Please check your API key and refresh the page." }],
                    chatHistory: [],
                    createdAt: Date.now()
                };
                setSessions([errorSession]);
                setActiveSessionId(errorSession.id);
                setStage('error');
            } finally {
                setIsLoading(false);
            }
        };
        initApp();
    }, [createNewSession, loadSessions]);

    useEffect(() => {
        if (ai && activeSession) {
            const newChat = ai.chats.create({
                model: 'gemini-2.5-flash',
                history: activeSession.chatHistory,
                config: { tools: [{ googleSearch: {} }] },
            });
            setChat(newChat);
        }
    }, [ai, activeSession]);

    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTo({
                top: chatContainerRef.current.scrollHeight,
                behavior: 'smooth'
            });
        }
    }, [renderedMessages]);
    
    const handleRemoveAttachment = useCallback(() => {
        setAttachment(prev => {
            if (prev) {
                URL.revokeObjectURL(prev.preview);
            }
            return null;
        });
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    }, []);

    const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            handleRemoveAttachment(); 
            const preview = URL.createObjectURL(file);
            setAttachment({ file, preview });
        }
    }, [handleRemoveAttachment]);

    const constructPrompt = useCallback((
        userMessageContent: string
    ): string => {
        const isMetaQuestion = stage === 'awaiting_task' && 
            /what is this app|explain your code|how do you work|your purpose|your features|your source code|who are you|what are you|self check|analyze your code|improve the app|make it better/i.test(userMessageContent);

        if (isMetaQuestion && sourceCode) {
            const redactedSourceCode = { ...sourceCode };
            if (redactedSourceCode['index.tsx']) {
                redactedSourceCode['index.tsx'] = redactedSourceCode['index.tsx'].replace(
                    /const MASTER_PROMPT = `[\s\S]*?`/,
                    `const MASTER_PROMPT = "[REDACTED FOR SELF-ANALYSIS]";`
                );
            }

            const encodedCodeContext = Object.entries(redactedSourceCode)
                .map(([fileName, content]) => `[FILE: ${fileName}]\n${btoa(unescape(encodeURIComponent(content)))}`)
                .join('\n\n');

            return `[META-QUESTION DETECTED] I will now operate under the SELF-AWARENESS PROTOCOL.\n\nHere are my Base64 encoded source files for analysis. You must decode them before answering my question.\n\n${encodedCodeContext}\n\nBased on the decoded source code, answer my question: ${userMessageContent}`;
        } else {
            const constraintsText = standingConstraints.length > 0
                ? `[Constraints]\n${standingConstraints.map(c => `- ${c}`).join('\n')}`
                : '[Constraints]\nNone.';
            
            const taskText = `[Task]\n${userMessageContent}`;

            return stage === 'awaiting_critique' ? userMessageContent : `${taskText}\n\n${constraintsText}`;
        }
    }, [standingConstraints, sourceCode, stage]);

    const updateSessionMessages = useCallback((
        sessionId: string,
        update: (messages: Message[]) => Message[]
    ) => {
        setSessions(prev => {
            const sessionIndex = prev.findIndex(s => s.id === sessionId);
            if (sessionIndex === -1) return prev;
            
            const newSessions = [...prev];
            const updatedSession = {
                ...newSessions[sessionIndex],
                messages: update(newSessions[sessionIndex].messages),
            };
            newSessions[sessionIndex] = updatedSession;
            return newSessions;
        });
    }, []);

    const handleSendMessage = useCallback(async (e: React.FormEvent) => {
        e.preventDefault();
        if (isLoading || !chat || !activeSession) return;

        let userMessageContent = '';
        if (stage === 'awaiting_task') {
            userMessageContent = userInput;
        } else if (stage === 'awaiting_critique') {
            userMessageContent = critique;
        } else {
            return;
        }

        const currentAttachment = attachment;
        const hasAttachment = currentAttachment !== null;
        if (!userMessageContent.trim() && !hasAttachment) return;

        setIsLoading(true);
        setStage('processing');

        const finalPromptToServer = constructPrompt(userMessageContent);

        const userMessage: Message = {
            id: Date.now(),
            author: 'user',
            content: userMessageContent,
            ...(hasAttachment && {
                attachment: {
                    preview: currentAttachment.preview,
                    type: currentAttachment.file.type,
                    name: currentAttachment.file.name,
                    size: currentAttachment.file.size
                }
            })
        };
        const aiMessagePlaceholder: Message = { id: Date.now() + 1, author: 'ai', content: '', citations: [] };

        updateSessionMessages(activeSession.id, (messages) => [...messages, userMessage, aiMessagePlaceholder]);

        setUserInput('');
        setCritique('');
        handleRemoveAttachment();

        try {
            let messageContent: string | any[];
            let userPartsForHistory: any[] = [];

            if (hasAttachment) {
                const base64Data = await fileToBase64(currentAttachment.file);
                const textPart = { text: finalPromptToServer };
                const imagePart = { inlineData: { mimeType: currentAttachment.file.type, data: base64Data } };
                
                messageContent = [textPart, imagePart];
                userPartsForHistory = [textPart, imagePart];

            } else {
                messageContent = finalPromptToServer;
                userPartsForHistory = [{ text: finalPromptToServer }];
            }

            const result = await chat.sendMessageStream({ message: messageContent });
            let fullResponseText = '';
            let citations: Citation[] = [];

            for await (const chunk of result) {
                fullResponseText += chunk.text;
                const newChunks = chunk.candidates?.[0]?.groundingMetadata?.groundingChunks;
                if (newChunks) {
                    citations = newChunks.map((c: any) => c.web).filter((c: any): c is Citation => c && c.uri && c.title);
                }

                updateSessionMessages(activeSession.id, (messages) => {
                    const newMessages = [...messages];
                    const lastMessageIndex = newMessages.length - 1;
                    if (lastMessageIndex >= 0 && newMessages[lastMessageIndex].id === aiMessagePlaceholder.id) {
                       newMessages[lastMessageIndex] = {
                           ...newMessages[lastMessageIndex],
                           content: fullResponseText,
                           citations,
                           parsedData: parseResponse(fullResponseText)
                       };
                    }
                    return newMessages;
                });
            }

            setSessions(prev => {
                const sessionIndex = prev.findIndex(s => s.id === activeSession.id);
                if (sessionIndex === -1) return prev;

                const newSessions = [...prev];
                const currentSession = newSessions[sessionIndex];

                const shouldRename = currentSession.messages.filter(m => m.author === 'user').length === 1 && currentSession.name === 'New Session';
                const newName = shouldRename ? userMessageContent.substring(0, 50) : currentSession.name;
                
                const userContent = { role: 'user', parts: userPartsForHistory };
                const modelContent = { role: 'model', parts: [{ text: fullResponseText }] };
                const newHistory = [...(currentSession.chatHistory || []), userContent, modelContent];
                
                newSessions[sessionIndex] = { ...currentSession, name: newName, chatHistory: newHistory };
                return newSessions;
            });

            if (finalPromptToServer.includes("[META-QUESTION DETECTED]")) {
                setStage('awaiting_task');
            } else if (fullResponseText.includes('[Draft]') && !fullResponseText.includes('[Final Output]')) {
                setStage('awaiting_critique');
            } else {
                setStage('awaiting_task');
            }

        } catch (error) {
            console.error("Message sending failed:", error);
            updateSessionMessages(activeSession.id, (messages) => {
                 const newMessages = [...messages];
                 const lastMessageIndex = newMessages.length - 1;
                 if (lastMessageIndex >= 0 && newMessages[lastMessageIndex].id === aiMessagePlaceholder.id) {
                    newMessages[lastMessageIndex] = {
                        ...newMessages[lastMessageIndex],
                        content: "[Error]\nAn unexpected error occurred while processing your request. Please try again. If the issue continues, please simplify your prompt or check the developer console for technical details."
                    };
                 }
                 return newMessages;
            });
            setStage('awaiting_task');
        } finally {
            setIsLoading(false);
        }
    }, [isLoading, chat, activeSession, userInput, critique, attachment, stage, sourceCode, standingConstraints, constructPrompt, updateSessionMessages, handleRemoveAttachment]);

    const handleTranslate = useCallback(async (messageId: number, textToTranslate: string, language: string) => {
        if (!ai || !activeSessionId) return;

        updateSessionMessages(activeSessionId, msgs => msgs.map(msg => msg.id === messageId ? { ...msg, isTranslating: true } : msg));

        try {
            const prompt = `Translate the following text to ${language}. Only provide the translated text, with no additional commentary or labels:\n\n"${textToTranslate}"`;
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: prompt,
            });

            updateSessionMessages(activeSessionId, msgs => msgs.map(msg =>
                msg.id === messageId
                    ? { ...msg, isTranslating: false, translation: { lang: language, content: response.text } }
                    : msg
            ));

        } catch (error) {
            console.error("Translation failed:", error);
             updateSessionMessages(activeSessionId, msgs => msgs.map(msg =>
                msg.id === messageId
                    ? { ...msg, isTranslating: false, translation: { lang: language, content: "Error: Translation failed." } }
                    : msg
            ));
        }
    }, [ai, activeSessionId, updateSessionMessages]);

    const handleExportLog = useCallback(() => {
        if (!activeSession) return;
        const logContent = activeSession.messages.map(msg => {
            const citations = msg.citations?.length ? `\n\n[CITATIONS]\n${msg.citations.map(c => `- ${c.title}: ${c.uri}`).join('\n')}` : '';
            const translation = msg.translation ? `\n\n[TRANSLATION (${msg.translation.lang})]\n${msg.translation.content}` : '';
            return `[${msg.author.toUpperCase()}]\n${msg.content}${citations}${translation}`;
        }).join('\n\n====================\n\n');

        const blob = new Blob([logContent], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        link.download = `zero-engine-log-${timestamp}.txt`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }, [activeSession]);

    const handleSaveConstraints = useCallback((newConstraints: string[]) => {
        setStandingConstraints(newConstraints);
        setIsConstraintsModalOpen(false);
    }, []);

    const handleNewSession = useCallback(async () => {
        if (!ai) return;
        setIsLoading(true);
        const newSessionId = await createNewSession(ai);
        setActiveSessionId(newSessionId);
        setIsSessionsModalOpen(false);
        setIsLoading(false);
    }, [ai, createNewSession]);

    const handleLoadSession = useCallback((sessionId: string) => {
        setActiveSessionId(sessionId);
        setIsSessionsModalOpen(false);
    }, []);

    const handleDeleteSession = useCallback((sessionId: string) => {
        setSessions(prev => {
            const remaining = prev.filter(s => s.id !== sessionId);
            if (remaining.length === 0) {
                localStorage.removeItem('zero_sessions');
                localStorage.removeItem('zero_active_session_id');
                window.location.reload(); 
                return [];
            }
            if (sessionId === activeSessionId) {
                setActiveSessionId(remaining[0].id); 
            }
            return remaining;
        });
    }, [activeSessionId]);

    const handleRenameSession = useCallback((sessionId: string, newName: string) => {
        setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, name: newName } : s));
    }, []);


    const getEngineStatus = useCallback((): EngineStatus => {
        switch (stage) {
            case 'initializing': return { label: 'Initializing', active: true };
            case 'awaiting_task': return { label: 'Awaiting Task', active: false };
            case 'processing': return { label: 'Processing...', active: true };
            case 'awaiting_critique': return { label: 'Awaiting Critique', active: true };
            case 'error': return { label: 'Engine Error', active: true };
            default: return { label: 'Standby', active: false };
        }
    }, [stage]);

    const status = getEngineStatus();

    return (
        <>
            <header>
                <div className="logo">
                    <svg className="logo-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="var(--primary-accent)" strokeWidth="1.5" />
                        <path d="M12 6V18" stroke="var(--primary-accent)" strokeWidth="1.5" strokeLinecap="round" />
                        <path d="M10 8L14 16" stroke="var(--primary-accent)" strokeWidth="1.5" strokeLinecap="round" />
                    </svg>
                    <h1>ZERO Execution Engine</h1>
                </div>
                <div className="header-actions">
                    <button className="action-btn header-action-btn" onClick={() => setIsSessionsModalOpen(true)} title="Manage Sessions"><SessionsIcon /></button>
                    <button className={`action-btn header-action-btn constraints-btn ${standingConstraints.length > 0 ? 'active' : ''}`} onClick={() => setIsConstraintsModalOpen(true)} title="Set Standing Constraints"><ConstraintsIcon /></button>
                    <button className="action-btn header-action-btn" onClick={() => setIsLicenseModalOpen(true)} title="License Information"><InfoIcon /></button>
                    <button className="action-btn header-action-btn" onClick={handleExportLog} title="Export Conversation Log"><ExportIcon /></button>
                    <div className={`engine-status ${status.active ? 'active' : ''}`}>{status.label}</div>
                </div>
            </header>
            <div className="chat-container" ref={chatContainerRef}>
                {isHistoryTruncated && (
                    <div className="history-notice">
                        Viewing the latest {MAX_RENDERED_MESSAGES} messages of this session. The full conversation is saved.
                    </div>
                )}
                {renderedMessages.map((msg) => (
                    <div key={msg.id} className={`chat-message ${msg.author}`}>
                        {msg.author === 'user' ? (
                            <UserMessageBlock content={msg.content} attachment={msg.attachment} />
                        ) : (
                            <AIMessageBlock message={msg} onTranslate={handleTranslate} />
                        )}
                    </div>
                ))}
                {isLoading && stage === 'processing' && (
                    <div className="chat-message ai">
                        <div className="loading-indicator">
                            <div className="atom">
                                <div className="atom-nucleus"></div>
                                <div className="atom-electron"></div>
                                <div className="atom-electron"></div>
                                <div className="atom-electron"></div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
            <div className="input-area">
                <form onSubmit={handleSendMessage}>
                    {attachment && (
                        <div className="attachment-preview">
                            {attachment.file.type.startsWith('image/') && (
                                <img src={attachment.preview} alt="Attachment preview" className="attachment-preview-image" />
                            )}
                            {attachment.file.type.startsWith('audio/') && (
                                <div className="attachment-preview-info">
                                    <span>{attachment.file.name}</span>
                                    <span>({(attachment.file.size / 1024).toFixed(2)} KB)</span>
                                </div>
                            )}
                            {attachment.file.type.startsWith('video/') && (
                                <video src={attachment.preview} muted playsInline className="attachment-preview-video" />
                            )}
                            {attachment.file.type === 'application/pdf' && (
                                <>
                                    <PdfIcon />
                                    <div className="attachment-preview-info">
                                        <span>{attachment.file.name}</span>
                                        <span>({(attachment.file.size / 1024).toFixed(2)} KB)</span>
                                    </div>
                                </>
                            )}
                            <button type="button" className="remove-attachment-btn" onClick={handleRemoveAttachment}><CloseIcon /></button>
                        </div>
                    )}
                    {stage === 'awaiting_task' && (
                        <>
                            <div className="input-group">
                                <label htmlFor="user-input">Your Request</label>
                                <div className="textarea-wrapper">
                                    <button type="button" className="attach-btn" title="Attach File" onClick={() => fileInputRef.current?.click()}><PaperclipIcon /></button>
                                    <textarea id="user-input" value={userInput} onChange={(e) => setUserInput(e.target.value)} placeholder="Enter your task, constraints, or a URL to analyze." disabled={isLoading}></textarea>
                                </div>
                            </div>
                            <div className="form-footer">
                                <button type="submit" disabled={isLoading || (!userInput.trim() && !attachment)}>Engage Engine</button>
                            </div>
                        </>
                    )}
                    {stage === 'awaiting_critique' && (
                        <>
                            <div className="input-group">
                                <label htmlFor="critique-input">Critique / Command</label>
                                <div className="textarea-wrapper">
                                    <button type="button" className="attach-btn" title="Attach File" onClick={() => fileInputRef.current?.click()}><PaperclipIcon /></button>
                                    <textarea id="critique-input" value={critique} onChange={(e) => setCritique(e.target.value)} placeholder="Provide corrections or type 'Proceed'." autoFocus disabled={isLoading}></textarea>
                                </div>
                            </div>
                            <div className="form-footer">
                                <button type="submit" disabled={isLoading || (!critique.trim() && !attachment)}>Submit Critique</button>
                            </div>
                        </>
                    )}
                    <input type="file" ref={fileInputRef} onChange={handleFileChange} style={{ display: 'none' }} accept="image/*,audio/*,video/*,application/pdf" />
                    {(stage === 'initializing' || stage === 'error') && <p style={{ textAlign: 'center', fontFamily: 'var(--font-mono)' }}>{status.label}</p>}
                </form>
            </div>
            {isConstraintsModalOpen && (
                <ConstraintsModal
                    isOpen={isConstraintsModalOpen}
                    onClose={() => setIsConstraintsModalOpen(false)}
                    onSave={handleSaveConstraints}
                    initialConstraints={standingConstraints}
                />
            )}
            {isSessionsModalOpen && (
                <SessionsModal
                    isOpen={isSessionsModalOpen}
                    onClose={() => setIsSessionsModalOpen(false)}
                    sessions={sessions}
                    activeSessionId={activeSessionId}
                    onLoadSession={handleLoadSession}
                    onNewSession={handleNewSession}
                    onDeleteSession={handleDeleteSession}
                    onRenameSession={handleRenameSession}
                />
            )}
            {isLicenseModalOpen && (
                <LicenseModal 
                    isOpen={isLicenseModalOpen}
                    onClose={() => setIsLicenseModalOpen(false)}
                />
            )}
        </>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(<React.StrictMode><App /></React.StrictMode>);
