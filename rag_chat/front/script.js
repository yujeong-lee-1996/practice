let currentMode = "chat";
let history = {
  chat: [],
  lie: [],
  rag: []
};

function switchMode(mode) {
  currentMode = mode;

  // 버튼 active 클래스 업데이트
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.getElementById(`mode-${mode}`).classList.add("active");

  renderChat();
}

async function send() {
  const msg = document.getElementById("msg").value;
  if (!msg.trim()) return;

  const res = await fetch(`/api/${currentMode}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: msg })
  });

  const data = await res.json();
  history[currentMode] = data.history;
  renderChat();
  document.getElementById("msg").value = "";
}

function renderChat() {
  const container = document.getElementById("chat-container");
  container.innerHTML = "";

  history[currentMode].forEach(({ role, message }) => {
    const div = document.createElement("div");
    div.className = role === "user" ? "user-msg" : "bot-msg";
    div.textContent = message;
    container.appendChild(div);
  });

  container.scrollTop = container.scrollHeight;
}
