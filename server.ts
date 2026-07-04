import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

// Initialize the Gemini SDK
// Note: User-Agent must be set to 'aistudio-build' for AI Studio metrics tracking.
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    }
  }
});

const app = express();
const PORT = 3000;

app.use(express.json());

// 1. Symptom Triage & Symptom Checker Endpoint
app.post("/api/gemini/symptom-check", async (req, res) => {
  try {
    const { symptoms, age, gender, village } = req.body;
    if (!symptoms) {
      return res.status(400).json({ error: "Symptoms are required" });
    }

    const systemInstruction = `You are an expert AI Triage Assistant in a Government Rural Primary Health Centre (PHC). 
Analyze the symptoms provided and output a highly reliable triage report in JSON format.
You must assess:
1. Possible underlying conditions (keep them clear, understandable, and culturally relevant to rural areas like water-borne, seasonal, or agricultural exposures).
2. Urgency level ('Low', 'Medium', 'High', 'Emergency').
3. Recommended hospital department ('General Medicine', 'Pediatrics', 'OB-GYN', 'Vaccination', 'Lab').
4. Suggested clinical priority token ('Normal', 'Emergency', 'Elderly', 'Maternal', 'Disabled') based on symptoms and patient stats (e.g., an infant or high fever is Emergency, a pregnant woman is Maternal).
5. Next steps / home care guidelines (translated into simple advice).
6. Local disease context advisory (e.g. if symptoms match local monsoon epidemics like Dengue, Cholera, or malaria).

Return ONLY a valid JSON object matching this structure:
{
  "possibleDiseases": ["string"],
  "urgencyLevel": "Low" | "Medium" | "High" | "Emergency",
  "recommendedDepartment": "General Medicine" | "Pediatrics" | "OB-GYN" | "Vaccination" | "Lab",
  "recommendedPriority": "Normal" | "Emergency" | "Elderly" | "Maternal" | "Disabled",
  "emergencyWarnings": ["string"],
  "homeCareAdvice": ["string"],
  "clinicalTriageNotes": "string",
  "isOutbreakRisk": boolean
}`;

    const prompt = `Patient Details:
Age: ${age || 'Unknown'} years old
Gender: ${gender || 'Unknown'}
Location/Village: ${village || 'Rural region'}
Symptom description: "${symptoms}"

Generate the rural clinical triage assessment in JSON format:`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        systemInstruction,
        responseMimeType: "application/json",
        temperature: 0.2,
      },
    });

    const responseText = response.text || "{}";
    const triageReport = JSON.parse(responseText.trim());
    res.json(triageReport);
  } catch (error: any) {
    console.error("Error in symptom-check API:", error);
    res.status(500).json({ 
      error: "Failed to generate symptom report", 
      details: error.message 
    });
  }
});

// 2. Doctor Treatment Helper Endpoint
app.post("/api/gemini/treatment-guide", async (req, res) => {
  try {
    const { patient, diagnosis, availableMedicines } = req.body;
    if (!patient || !diagnosis) {
      return res.status(400).json({ error: "Patient and diagnosis details are required" });
    }

    const systemInstruction = `You are a Senior Medical Specialist and Clinical Decision Support System helping rural Primary Health Centre (PHC) doctors.
Your goal is to suggest a safe, effective treatment plan based on the patient's vitals, allergies, history, and the diagnosis.
CRITICAL MANDATE: Recommend medicines PRIMARILY from the list of 'AVAILABLE MEDICINES' in the PHC pharmacy. Rural patients cannot afford external private pharmacies. If an essential medicine is unavailable, explicitly state it as 'NOT IN STOCK - MUST PROCURE/REFER' and recommend an alternative.

Provide clinical advice in a professional JSON format matching:
{
  "treatmentRationale": "string summarizing medical rationale",
  "recommendedMedicines": [
    { "name": "string", "dosage": "string", "duration": "string", "instructions": "string (e.g. after meals)", "isAvailable": boolean }
  ],
  "homeAdvice": ["string"],
  "warningSigns": ["string"],
  "referralNeeded": boolean,
  "referralReason": "string (if needed, else empty)",
  "followUpDays": number
}`;

    const prompt = `Patient Profile:
Name: ${patient.name}
Age: ${patient.age} | Gender: ${patient.gender}
Village: ${patient.village}
Vitals: BP ${patient.vitals?.bp || "N/A"}, Temp ${patient.vitals?.temp || "N/A"}, Pulse ${patient.vitals?.pulse || "N/A"}, Blood Sugar ${patient.vitals?.bloodSugar || "N/A"}
Allergies: ${patient.allergies?.join(", ") || "None reported"}
Medical History: ${patient.medicalHistory?.join(", ") || "None reported"}
Chronic Diseases: ${patient.chronicDiseases?.join(", ") || "None reported"}

Diagnosis: "${diagnosis}"

Available PHC Medicines in Pharmacy:
${JSON.stringify(availableMedicines || [])}

Generate the clinical recommendation:`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        systemInstruction,
        responseMimeType: "application/json",
        temperature: 0.1,
      },
    });

    const responseText = response.text || "{}";
    const treatmentGuide = JSON.parse(responseText.trim());
    res.json(treatmentGuide);
  } catch (error: any) {
    console.error("Error in treatment-guide API:", error);
    res.status(500).json({ 
      error: "Failed to generate treatment recommendations", 
      details: error.message 
    });
  }
});

// 3. Inventory Procurement Forecasting
app.post("/api/gemini/inventory-forecast", async (req, res) => {
  try {
    const { inventory, seasonalityNotes } = req.body;
    if (!inventory) {
      return res.status(400).json({ error: "Inventory list is required" });
    }

    const systemInstruction = `You are an AI Healthcare Logistics and Supply Chain Optimizer for Government National Health Missions.
Analyze the current stock, daily consumption, batch expiries, and reorder levels of various essential drugs.
Provide a procurement forecast in JSON format.
For each item in the inventory, suggest:
1. "forecastProcurementQty": Estimated quantity to order to prevent stock-outs over the next 60 days, incorporating current stock, daily consumption rates, and seasonality.
2. "riskLevel": 'Safe' | 'Low Stock Risk' | 'Expiry Risk' | 'Stockout Imminent'.
3. "justification": A clear, single-sentence supply-chain explanation.

Output a JSON array representing the updated medicine analysis:
[
  {
    "id": "string",
    "name": "string",
    "forecastProcurementQty": number,
    "riskLevel": "Safe" | "Low Stock Risk" | "Expiry Risk" | "Stockout Imminent",
    "justification": "string"
  }
]`;

    const prompt = `Current Inventory Data:
${JSON.stringify(inventory)}

Seasonality & Disease Exposure Context:
"${seasonalityNotes || "Monsoon season approaching, potential surge in waterborne diseases, malaria, and dengue."}"

Generate the supply-chain forecast:`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        systemInstruction,
        responseMimeType: "application/json",
        temperature: 0.2,
      },
    });

    const responseText = response.text || "[]";
    const forecast = JSON.parse(responseText.trim());
    res.json(forecast);
  } catch (error: any) {
    console.error("Error in inventory-forecast API:", error);
    res.status(500).json({ 
      error: "Failed to calculate inventory forecast", 
      details: error.message 
    });
  }
});

// 4. District Outbreak Predictions & Heatmap Analytics
app.get("/api/gemini/outbreak-predictions", async (req, res) => {
  try {
    const systemInstruction = `You are a WHO Disease Surveillance specialist and Public Health AI Epidemiologist.
Predict potential seasonal disease outbreaks and high-risk health metrics for 4 fictional rural districts of Maharashtra, India based on current seasonal patterns, low sanitation infrastructure reports, and patient complaints.

Output a JSON array of outbreak predictions with exact structures:
[
  {
    "district": "string (e.g. Pune, Nagpur, Gadchiroli, Nashik)",
    "disease": "string (e.g. Malaria, Cholera, Dengue, Japanese Encephalitis, Malnutrition)",
    "riskScore": number (0 to 100),
    "predictedCases": number,
    "confidence": number (percentage e.g. 85),
    "timeline": "string (e.g. Next 2-4 weeks)",
    "recommendations": ["string (public health actions required)"]
  }
]

Create a highly realistic epidemiology report with 4 items, one for each district. Ensure Pune has a risk for Cholera (monsoon flood season), Gadchiroli has a risk for Malaria (heavy forest-fringe vector breeding), Nagpur has a risk for Dengue, and Nashik has a risk for Malnutrition.`;

    const prompt = "Generate the 4-district seasonal outbreak risk forecast in JSON format for Maharashtra:";

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        systemInstruction,
        responseMimeType: "application/json",
        temperature: 0.3,
      },
    });

    const responseText = response.text || "[]";
    const predictions = JSON.parse(responseText.trim());
    res.json(predictions);
  } catch (error: any) {
    console.error("Error in outbreak-predictions API:", error);
    // Fallback static data if Gemini fails
    res.json([
      {
        district: "Pune",
        disease: "Cholera & Acute Diarrhea",
        riskScore: 82,
        predictedCases: 140,
        confidence: 88,
        timeline: "Next 2 weeks (due to heavy monsoon runoff and drinking water contamination)",
        recommendations: ["Distribute chlorine tablets", "Establish emergency rehydration camps", "Launch community sanitation drives"]
      },
      {
        district: "Gadchiroli",
        disease: "Malaria (Plasmodium falciparum)",
        riskScore: 78,
        predictedCases: 210,
        confidence: 85,
        timeline: "Next 4 weeks (monsoon forest-canopy vector breeding cycle)",
        recommendations: ["Distribute insecticide-treated bed nets", "Conduct thermal fogging in tribal hamlets", "Stock Artemisinin Combination Therapy (ACT) medicines"]
      },
      {
        district: "Nagpur",
        disease: "Dengue Fever",
        riskScore: 65,
        predictedCases: 95,
        confidence: 80,
        timeline: "Next 3 weeks (water accumulation in peri-urban and rural containers)",
        recommendations: ["Source reduction of vector breeding sites", "Deploy community health workers for dry-day campaigns", "Enable platelet count checks at PHCs"]
      },
      {
        district: "Nashik",
        disease: "Severe Acute Malnutrition (SAM)",
        riskScore: 70,
        predictedCases: 320,
        confidence: 90,
        timeline: "Continuous (critical pre-harvest lean period in tribal blocks)",
        recommendations: ["Initiate spot-feeding centers", "Intensify Poshan Abhiyaan nutritional monitoring", "Deliver supplementary therapeutic foods"]
      }
    ]);
  }
});

// Setup Express and Vite Middleware integration
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    console.log("Starting server in development mode with Vite...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("Starting server in production mode...");
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Rural Healthcare platform listening on port ${PORT}`);
  });
}

startServer();
