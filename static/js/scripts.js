// Obtener el token CSRF
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function getCSRFToken() {
    return getCookie('csrftoken');
}

function appendMessage(content, sender, isMarkdown = false) {
    const chatHistory = document.getElementById("chatHistory");

    const messageElement = document.createElement("div");
    messageElement.classList.add("chat-message", sender);

    // Si el contenido es Markdown, conviértelo a HTML usando marked.js
    if (isMarkdown) {
        messageElement.innerHTML = marked.parse(content); // Convertir Markdown a HTML
    } else {
        messageElement.textContent = content; // Añadir texto normal
    }

    chatHistory.appendChild(messageElement);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Desplazarse hacia abajo automáticamente
}

// Mostrar animación de "Escribiendo..." con puntos animados
function showTypingIndicator() {
    const chatHistory = document.getElementById("chatHistory");

    const typingIndicator = document.createElement("div");
    typingIndicator.id = "typingIndicator";
    typingIndicator.classList.add("chat-message", "bot");

    // Crear contenedor para los puntos animados
    const dotsContainer = document.createElement("span");
    dotsContainer.classList.add("dots-container");

    // Crear los puntos animados
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("span");
        dot.classList.add("dot");
        dot.textContent = ".";
        dotsContainer.appendChild(dot);
    }

    typingIndicator.textContent = "Escribiendo "; // Texto antes de los puntos
    typingIndicator.appendChild(dotsContainer); // Añadir puntos animados

    chatHistory.appendChild(typingIndicator);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Ocultar animación de "Escribiendo..."
function hideTypingIndicator() {
    const typingIndicator = document.getElementById("typingIndicator");
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Bloquear el botón de enviar
function disableSendButton() {
    const sendButton = document.querySelector(".chat-input button");
    sendButton.disabled = true;
    sendButton.style.backgroundColor = "#ccc"; // Cambiar color para indicar que está deshabilitado
    sendButton.style.cursor = "not-allowed"; // Cambiar cursor para indicar que no es clicable
}

// Desbloquear el botón de enviar
function enableSendButton() {
    const sendButton = document.querySelector(".chat-input button");
    sendButton.disabled = false;
    sendButton.style.backgroundColor = "#002855"; // Restaurar el color original
    sendButton.style.cursor = "pointer"; // Restaurar el cursor original
}

// Función para enviar el mensaje
function sendMessage() {
    const userInput = document.getElementById("userMessage");
    const message = userInput.value.trim();

    if (message === "") return; // No enviar mensajes vacíos

    // Añadir el mensaje del usuario al chat
    appendMessage(message, "user");

    // Mostrar el indicador de "Escribiendo..." y Bloquear el botón de enviar
    showTypingIndicator();
    disableSendButton();

    // Preparar datos para enviar
    const data = { message: message };

    fetch("/api/send_message/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCSRFToken(),
        },
        body: JSON.stringify(data),
    })
        .then((response) => response.json())
        .then((data) => {
            // Ocultar el indicador de "Escribiendo..." y Desbloquear el botón de enviar
            hideTypingIndicator();
            enableSendButton();
            
            if (data.response) {
                // Añadir la respuesta del bot al chat con Markdown convertido a HTML
                appendMessage(data.response, "bot", true);
            } else if (data.error) {
                // Manejar errores devueltos por el servidor
                appendMessage("Error: " + data.error, "bot");
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            hideTypingIndicator();
            appendMessage("Error al comunicarse con el servidor.", "bot");
        });

    userInput.value = ""; 
}

// Enviar mensaje al presionar Enter
document.getElementById("userMessage").addEventListener("keyup", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});