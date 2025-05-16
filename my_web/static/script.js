let currentMode = "chat";

// 전역 상태 관리 
let history = {
  chat: [],
  lie: [],
  rag: []
};

// 모드 전환 함수 
function switchMode(mode) {
  currentMode = mode;

  // 버튼 active 클래스 업데이트
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.getElementById(`mode-${mode}`).classList.add("active");

  renderChat();
}

async function send() {
  const msg = document.getElementById("msg").value;  // 사용자가 입력한 내용 
  if (!msg.trim()) return;

  const res = await fetch(`/project/chatbot/${currentMode}`, { 
    method: "POST", // 백엔드로 POST 요청 
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: msg })
  });

  const data = await res.json(); // 서버로부터 반환받은 값을 히스토리에 반영 
  history[currentMode] = data.history;
  renderChat();
  document.getElementById("msg").value = "";
}

// 채팅 UI 렌더링 함수 
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
