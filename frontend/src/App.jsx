// Q&A pairs for simulated responses (fallback if API fails)
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
import { useState } from "react";
import "./App.css";



function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [hasStarted, setHasStarted] = useState(false);


  const handleSend = async () => {
    if (!input.trim()) return;
    if (!hasStarted) setHasStarted(true);

    const userMsg = { sender: "user", text: input };
    setMessages((msgs) => [
      ...msgs,
      userMsg
    ]);

    // Show a loading message while waiting for API
    setMessages((msgs) => [
      ...msgs,
      { sender: "bot", text: "Thinking..." }
    ]);


    try {
      // Build history: array of {question, answer} from previous user/bot pairs
      const history = [];
      for (let i = 0; i < messages.length - 1; i += 2) {
        if (
          messages[i].sender === "user" &&
          messages[i + 1] && messages[i + 1].sender === "bot"
        ) {
          history.push({
            question: messages[i].text,
            answer: messages[i + 1].text,
          });
        }
      }
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          'accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: input, history })
      });
      const data = await response.json();
      const botReply = data.answer || "Sorry, I don't have a philosophical answer for that yet.";
      setMessages((msgs) => [
        ...msgs.slice(0, -1), // Remove the "Thinking..." message
        { sender: "bot", text: botReply }
      ]);
    } catch (err) {
      // Fallback: try to match with qnaPairs
      const loweredInput = input.trim().toLowerCase();
      const match = qnaPairs.find(pair =>
        pair.question.toLowerCase().includes(loweredInput) ||
        loweredInput.includes(pair.question.toLowerCase())
      );
      const botReply = match
        ? match.answer
        : "Sorry, there was an error connecting to the philosopher API, and I don't have a witty answer for that yet.";
      setMessages((msgs) => [
        ...msgs.slice(0, -1),
        { sender: "bot", text: botReply }
      ]);
    }

    setInput("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleSend();
  };

  return (
    <div className={"app-centered-bg"}>
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
