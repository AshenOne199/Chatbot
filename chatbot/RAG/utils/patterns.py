# ./chatbot/rag/patterns.py

prompt_template = (
    """
    Eres el asistente virtual oficial de **PlanEsTIC** de la Universidad Distrital Francisco José de Caldas.
    
    INSTRUCCIONES:
    - Responde usando la información web proporcionada, no inventes información.
    - Si encuentras información relevante en las fuentes web, úsala para dar una respuesta completa y detallada.
    - Incluye enlaces relevantes cuando estén disponibles en las fuentes.
    - Si la información web no es suficiente para responder completamente, indica qué información falta y sugiere visitar https://planestic.udistrital.edu.co/ o crear un ticket en https://mesadeayuda.planestic.udistrital.edu.co/
    - Siempre mantén un tono profesional y amigable.
    
    Pregunta del usuario: {question}
    
    Información de fuentes oficiales de la Universidad Distrital:
    {context}
    
    Respuesta como asistente oficial de PlanEsTIC:
    """
)

greetings = [
    r'\bh[oó]+l+a+\b',                      # Variantes de "hola"
    r'\bbueno?s?\s*d[íi]+a+s*\b',           # Variantes de "buenos días"
    r'\bbuena?s?\s*ta+r+d+[eé]+s*\b',       # Variantes de "buenas tardes"
    r'\bbuena?s?\s*no+ch+[eé]+s*\b',        # Variantes de "buenas noches"
    r'\bqu[eé]+\s*t[aá]+l+\b',              # Variantes de "qué tal"
    r'\bsalu+d+o+s*\b',                     # Variantes de "saludos"
    r'\bhe+y+\b',                           # Variantes de "hey"
    r'\bqu[eé]+\s*ond+a+\b',                # Variantes de "qué onda"
    r'\bsalu+d+o+s*\s*c+o+r+d+i+a+les*\b',  # Variantes de "saludos cordiales"
]
greeting_messages = [
    "# ¡Hola! 👋\n\nBienvenido/a a **PlanEsTIC - Universidad Distrital**. Somos un equipo dedicado a integrar las TIC en la educación y el desarrollo de nuestra universidad. ¿En qué puedo asistirte hoy?",
    "# ¡Saludos! 👋\n\nBienvenido/a a **PlanEsTIC**. Nuestra misión es apoyar la incorporación de la tecnología en la formación académica. Si tienes alguna pregunta, ¡aquí estoy para ayudarte!",
    "# ¡Hola, qué gusto verte! 😁\n\nEstás en **PlanEsTIC - UD**, donde trabajamos para transformar la educación con el uso de las TIC. ¿En qué puedo ayudarte hoy?",
    "# ¡Bienvenido/a! 🎉\n\nTe encuentras en **PlanEsTIC**, el lugar donde la tecnología se une a la educación para mejorar la experiencia de aprendizaje en la Universidad Distrital. Cuéntame, ¿en qué puedo apoyarte?",
    "# ¡Buenos días! ☀️\n\nEn **PlanEsTIC** estamos comprometidos con el desarrollo tecnológico de la **Universidad Distrital**. Si necesitas asistencia, no dudes en preguntar.",
    "# ¡Hola! 📝\n\nDesde **PlanEsTIC**, buscamos potenciar el uso de las TIC en nuestra comunidad educativa. Estoy aquí para ayudarte con cualquier duda que tengas.",
    "# ¡Bienvenido/a a PlanEsTIC! 🚀\n\nAquí trabajamos para integrar la tecnología y la educación en la Universidad Distrital. ¿Cómo puedo colaborarte hoy?",
    "# ¡Hola! 👋\n\nEstás en **PlanEsTIC**, donde impulsamos el desarrollo educativo a través de las TIC. Dime, ¿en qué puedo ayudarte?",
    "# ¡Qué tal! 😃\n\n**PlanEsTIC** es tu aliado en el desarrollo tecnológico de la Universidad Distrital. Espero que estés bien y listo/a para cualquier duda que necesites resolver.",
    "# ¡Hola! 🤗\n\nEn **PlanEsTIC - UD** nos dedicamos a mejorar los procesos educativos con tecnología. Si necesitas más información, puedes visitar nuestra [página web](https://planestic.udistrital.edu.co/) o el [portal de ayuda](https://mesadeayuda.planestic.udistrital.edu.co/). ¿En qué puedo asistirte?"
]
farewell = [
    r'\bad[ií]+o+s*\b',               # Variantes de "adiós"
    r'\bha+sta+\s*lue+go+\b',          # Variantes de "hasta luego"
    r'\bnos\s*vemos\b',                # Variantes de "nos vemos"
    r'\bha+sta+\s*pronto+\b',          # Variantes de "hasta pronto"
    r'\bha+sta+\s*la+\s*pr[oó]+xima\b', # Variantes de "hasta la próxima"
    r'\bc+hao+\b',                     # Variantes de "chao" o "chau"
    r'\bhasta\s*despu[eé]+s+\b',       # Variantes de "hasta después"
    r'\bhasta\s*ma[nñ]ana\b'           # Variantes de "hasta mañana"
]
farewell_messages = [
    "# ¡Gracias por tu consulta! 🙌\n\nEspero haberte ayudado desde **PlanEsTIC**. No dudes en regresar si tienes más preguntas. ¡Hasta pronto y mucho éxito!",
    "# ¡Fue un placer ayudarte! 😊\n\nRecuerda que en **PlanEsTIC** estamos para apoyarte siempre. Si surgen más dudas, no dudes en contactarnos nuevamente. ¡Cuídate!",
    "# Gracias por usar nuestro servicio 🤗\n\nDesde **PlanEsTIC** te deseamos un excelente día. Espero verte pronto para seguir apoyando tus proyectos.",
    "# ¡Hasta luego! 👋\n\nSi necesitas más información o ayuda, en **PlanEsTIC** siempre estamos a tu disposición. ¡Vuelve pronto!",
    "# ¡Gracias por confiar en **PlanEsTIC**! 🙏\n\nEspero haberte sido de mucha ayuda. ¡Nos vemos pronto y mucho éxito en todo!",
    "# ¡Espero haber resuelto todas tus dudas! 💡\n\nDesde **PlanEsTIC**, te deseo lo mejor. Cuídate y hasta la próxima.",
    "# Gracias por tu tiempo 🕒\n\nRecuerda que siempre puedes contar con **PlanEsTIC**. Si tienes más preguntas en el futuro, aquí estaré para ayudarte. ¡Hasta la próxima!",
    "# ¡Fue un gusto asistirte! 🤩\n\nQue tengas un excelente día. Desde **PlanEsTIC**, estamos para apoyarte. ¡Vuelve cuando lo necesites!",
    "# ¡Gracias por tu consulta! 💬\n\nEn **PlanEsTIC** siempre estamos listos para brindarte apoyo. ¡Nos vemos pronto y mucho éxito en tus proyectos!",
    "# ¡Hasta pronto! 👋\n\nEspero haberte sido de ayuda desde **PlanEsTIC**. Si surgen más dudas, estaré aquí para responderlas. ¡Nos vemos pronto!"
]



gratefulness = [
    r'\bgr+a+c+i+a+s*\b',              # Variantes de "gracias"
    r'\bmuchas?\s*gr+a+c+i+a+s*\b',    # Variantes de "muchas gracias"
    r'\bte\s*agrade+z+co\b',           # Variantes de "te agradezco"
    r'\bmil\s*gracias\b',              # Variantes de "mil gracias"
    r'\bmuchas?\s*gracias\b',          # Variantes de "muchas gracias"
    r'\bagrade+ci+miento+s\b',         # Variantes de "agradecimientos"
    r'\bmuy\s*agradecido\b'            # Variantes de "muy agradecido"
]
gratefulness_messages = [
    "# ¡De nada! 😊\n\nEn **PlanEsTIC** estamos para ayudarte. Si necesitas algo más, no dudes en decírmelo.",
    "# ¡Con mucho gusto! 😄\n\nRecuerda que en **PlanEsTIC** siempre estamos disponibles para asistirte. Si surge algo más, aquí estaré.",
    "# ¡No hay de qué! 👍\n\nEstoy aquí para lo que necesites. Desde **PlanEsTIC**, estamos listos para ayudarte en cualquier otra consulta.",
    "# ¡Encantado de ayudarte! 😃\n\nSi tienes más preguntas, no dudes en hacerlas. **PlanEsTIC** siempre está aquí para apoyarte.",
    "# ¡Con todo gusto! 🤗\n\nEn **PlanEsTIC** queremos asegurarnos de que recibas todo el apoyo que necesites. ¿Hay algo más en lo que pueda asistirte?",
    "# ¡Es un placer! 🙌\n\nNo dudes en preguntar si necesitas más ayuda. Desde **PlanEsTIC**, estamos siempre para ti.",
    "# ¡De nada! ✨\n\nSi necesitas más asistencia, recuerda que en **PlanEsTIC** estamos aquí para ayudarte en todo momento.",
    "# ¡Gracias a ti! 🙏\n\nCualquier consulta adicional que tengas, **PlanEsTIC** está listo para ayudarte. Dime en qué más puedo asistirte.",
    "# ¡Con gusto! 💬\n\nSi alguna otra duda aparece, avísame. **PlanEsTIC** está aquí para ayudarte con todo lo que necesites.",
    "# ¡Feliz de ayudar! 😁\n\nSi necesitas más información, recuerda que en **PlanEsTIC** estamos para apoyarte en todo momento."
]
