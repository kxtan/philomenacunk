import { useState } from "react";
import "./App.css";

function App() {
  const [input, setInput] = useState("");

  const handleSend = () => {
    alert(`You entered: ${input}`);
    setInput("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleSend();
  }; 

  return (
    <div className="app-centered-bg">
      <div className="app-brand">
        <span className="app-brand__title">philomenacunk</span>
        <div className="app-brand__desc">
          not inspired by philosophy and history
        </div>
      </div>

      <div className="app-float-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything..."
        />
        <button className="app-sendbtn" onClick={handleSend} title="Send">
          <svg fill="none" viewBox="0 0 24 24">
            <path d="M2 21l21-9-21-9v7l15 2-15 2v7z" fill="#000000ff"/>
          </svg>
        </button>
      </div>
    </div>
  );
}

export default App;
