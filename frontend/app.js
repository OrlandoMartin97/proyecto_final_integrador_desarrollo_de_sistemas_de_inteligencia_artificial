async function predecir() {

    let data = {
        age: document.getElementById("age").value,
        admission_grade: document.getElementById("admission_grade").value,
        scholarship: document.getElementById("scholarship").value,
        approved: document.getElementById("approved").value,
        avg_grade: document.getElementById("avg_grade").value
    };

    try {
        let response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        let result = await response.json();

        // Determinar emoji según la predicción
        let emoji = "🎓";
        if (result.prediccion.includes("Deserción") || result.prediccion.includes("riesgo alto")) {
            emoji = "⚠️";
        } else if (result.prediccion.includes("activo") || result.prediccion.includes("riesgo medio")) {
            emoji = "📚";
        } else if (result.prediccion.includes("Graduado") || result.prediccion.includes("bajo riesgo")) {
            emoji = "🎉";
        }

        const mensajeHTML = `
            <div style="font-size: 40px; margin-bottom: 15px;">${emoji}</div>
            <div style="font-size: 20px; font-weight: 700; color: #667eea; margin-bottom: 10px;">
                ${result.prediccion}
            </div>
            <div style="font-size: 16px; color: #666;">
                Confianza: <strong style="color: #764ba2;">${result.confianza}</strong>
            </div>
        `;

        mostrarResultado(mensajeHTML, "success");

    } catch (error) {
        mostrarResultado("❌ Error al conectar con el servidor. Verifica que Flask esté corriendo.", "error");
        console.error("Error:", error);
    }
}

function mostrarResultado(mensaje, tipo) {
    const resultadoContainer = document.getElementById("resultado-container");
    const resultado = document.getElementById("resultado");
    
    resultado.innerHTML = mensaje;
    resultadoContainer.classList.remove("hidden");
    
    // Scroll suave hacia el resultado
    resultadoContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
}