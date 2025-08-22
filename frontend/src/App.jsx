import { useState } from "react";
import "./App.css";

// Q&A pairs for simulated responses
const qnaPairs = [
  {
    question: "Why is adulting so hard?",
    answer:
      "Because, dear mortal, Plato never wrote “The Republic of Paying Bills.” He left that part out, assuming the philosopher-king’s subjects would get free Wi-Fi and eternal health insurance. Spoiler: they didn’t.",
  },
  {
    question: "Should I quit my job and pursue my passion?",
    answer:
      "Plato would say your “true form” lies beyond the cave. But remember: in the cave, rent is due on the 1st. Passion is great, but so is air conditioning.",
  },
  {
    question: "Why do people ghost each other?",
    answer:
      "Aristotle would call humans “social animals.” Apparently, some are more “ghostly animals.” Plato might suggest they’re stuck in the shadows, afraid of facing the blinding light of… accountability.",
  },
  {
    question: "Is happiness real, or just an illusion?",
    answer:
      "Plato would say happiness is the pursuit of the eternal Good. Epicurus would say it’s a cheese platter with wine. Honestly, the cheese guy seems more practical.",
  },
  {
    question: "Why do I keep procrastinating?",
    answer:
      "Because Socrates told you to “know thyself.” And now you know thyself prefers scrolling memes over doing thy taxes. Congratulations, you’re a philosopher.",
  },
  {
    question: "Does love really exist?",
    answer:
      "Plato described love as the soul’s longing for its missing half. But then Tinder invented “Super Likes,” which really ruined the mystique.",
  },
  {
    question: "Why is life unfair?",
    answer:
      "Life was never fair. Ask Diogenes—he lived in a barrel. If fairness was a thing, he’d at least have gotten a studio apartment with running water.",
  },
  {
    question: "How do I find meaning in life?",
    answer:
      "Nietzsche would say: “Create your own meaning.” Plato would say: “Seek the ideal forms.” I’d say: maybe start by finishing your laundry.",
  },
  {
    question: "Why do I feel stuck in life?",
    answer:
      "Because you’re literally living Plato’s Allegory of the Cave—staring at shadows (aka Netflix suggestions) instead of walking outside. Try sunlight; it’s free.",
  },
  {
    question: "What’s the secret to a good life?",
    answer:
      "Socrates said, “The unexamined life is not worth living.” But he didn’t have Spotify, so maybe chill—examining your playlists counts too.",
  },
];

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [hasStarted, setHasStarted] = useState(false);

  const handleSend = () => {
    if (!input.trim()) return;
    if (!hasStarted) setHasStarted(true);

    setMessages((msgs) => [
      ...msgs,
      { sender: "user", text: input }
    ]);

    setTimeout(() => {
      // Loose matching
      const loweredInput = input.trim().toLowerCase();
      const match = qnaPairs.find(pair =>
        pair.question.toLowerCase().includes(loweredInput) ||
        loweredInput.includes(pair.question.toLowerCase())
      );
      const botReply = match
        ? match.answer
        : `Sorry, I don't have a philosophical answer for that yet.`;

      setMessages((msgs) => [
        ...msgs,
        { sender: "bot", text: botReply }
      ]);
    }, 500);

    setInput("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleSend();
  };

  return (
    <div className={`app-centered-bg ${hasStarted ? "app-centered-bg--chat" : "app-centered-bg--centered"}`}>
      <div className={`app-brand ${hasStarted ? "app-brand--top" : "app-brand--centered"}`}>
        <span className="app-brand__title">philomenacunk</span>
        <div className="app-brand__desc">
          not inspired by philosophy and history
        </div>
      </div>

      <div className="app-message-list">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={msg.sender === "user" ? "app-msg-user" : "app-msg-bot"}
          >
            {msg.text}
          </div>
        ))}
      </div>

      <div className={`app-float-input ${hasStarted ? "app-float-input--bottom" : "app-float-input--centered"}`}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything..."
        />
        <button className="app-sendbtn" onClick={handleSend} title="Send">
          <svg fill="none" viewBox="0 0 24 24">
            <path d="M2 21l21-9-21-9v7l15 2-15 2v7z" fill="#000000ff" />
          </svg>
        </button>
      </div>
    </div>
  );
}

export default App;
