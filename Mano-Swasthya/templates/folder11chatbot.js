async function sendMessage() {
    let userInput = document.getElementById("userInput").value;
    if (!userInput) return;

    let chatBox = document.getElementById("chatBody");

    // Add user message to chat
    let userMessage = document.createElement("div");
    userMessage.className = "message user";
    userMessage.innerText = "You: " + userInput;
    chatBox.appendChild(userMessage);

    document.getElementById("userInput").value = "";

    // Send message to Flask backend
    let response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userInput })
    });

    let data = await response.json();

    // Add bot response to chat
    let botMessage = document.createElement("div");
    botMessage.className = "message bot";
    botMessage.innerText = "Bot: " + data.response;
    chatBox.appendChild(botMessage);

    chatBox.scrollTop = chatBox.scrollHeight;
}