import express from 'express'
import cors from 'cors';
// import fetch from 'node-fetch';
import { PythonShell } from 'python-shell';
import path from 'path';
import { dirname } from 'path';
import { fileURLToPath } from 'url';
const app = express();
app.use(express.json());  // to parse JSON requests
app.use(cors({
  origin:'http://localhost:5173',
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
allowedHeaders: ['Content-Type', 'Authorization'],
  credentials:true,
}))
// Endpoint for making predictions
app.post('/analyze', async (req, res) => {
  
  console.log(req.body)
  try {
    // Send the data to the Flask API using fetch
    const response = await fetch('http://127.0.0.1:5020/main', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        
      },
      body: JSON.stringify(req.body), // Send the data as JSON
    });
    
      // Check if the response is successful
      if (!response.ok) {
          throw new Error('Error with Flask API response');
      }

      // Parse the response body as JSON
      const data = await response.json();

      // Send the response from Flask back to the client
      res.json(data);
  } catch (error) {
      console.error(error);
      res.status(500).json({ 
        status: "error",
        message: "Error connecting to Flask API",
        error: error.message
    });
  }
});

const PORT = 5010;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});