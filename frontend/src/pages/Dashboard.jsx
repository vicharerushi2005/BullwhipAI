import { useEffect, useState } from "react";
import ProbabilityChart from "../components/ProbabilityChart";

import {
  AlertTriangle,
  Database,
  Package,
  MapPin,
  BarChart3,
  Activity
} from "lucide-react";

import StatCard from "../components/StatCard";

import {
  getPrediction,
  getSummary,
  getExplanation,
  getRecommendation,
  getInventory
} from "../api/api";

function Dashboard() {

  const [prediction, setPrediction] = useState(null);
  const [summary, setSummary] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [inventory, setInventory] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {

    const [
        pred,
        sum,
        exp,
        rec,
        inv
    ] = await Promise.all([
        getPrediction(),
        getSummary(),
        getExplanation(),
        getRecommendation(),
        getInventory()
    ]);

    setPrediction(pred);
    setSummary(sum);
    setExplanation(exp);
    setRecommendation(rec);
    setInventory(inv);
}

  if (
    !prediction ||
    !summary ||
    !explanation ||
    !recommendation ||
    !inventory
) {
    return (
      <h1 style={{ color: "white" }}>
        Loading BullwhipAI...
      </h1>
    );
  }

  return (

    <div>

      <h1>🚀 BullwhipAI Dashboard</h1>

      <p className="subtitle">
        Autonomous Multi-Agent Supply Chain Intelligence System
      </p>

      <div className="grid">

        <StatCard
          title="Current Risk"
          value={prediction.prediction}
          color="#facc15"
          icon={<AlertTriangle size={34} />}
        />

        <StatCard
          title="Confidence"
          value={`${prediction.confidence}%`}
          color="#38bdf8"
          icon={<Activity size={34} />}
        />

        <StatCard
          title="Dataset Records"
          value={summary.total_records}
          color="#4ade80"
          icon={<Database size={34} />}
        />

      </div>

      <div className="grid">

        <StatCard
          title="Products"
          value={summary.products}
          color="#fb923c"
          icon={<Package size={34} />}
        />

        <StatCard
          title="Cities"
          value={summary.cities}
          color="#c084fc"
          icon={<MapPin size={34} />}
        />

        <StatCard
          title="Latest Data"
          value={summary.latest_date}
          color="#22d3ee"
          icon={<BarChart3 size={34} />}
        />

      </div>

  <div className="section">

<h2>
📊 AI Confidence Distribution
</h2>

<br/>

<ProbabilityChart
probabilities={prediction.probabilities}
/>

</div>

      <div className="section">

        <h2>
          🤖 AI Recommendation
        </h2>

        <br />

        {prediction.prediction === "High" && (
          <>
            <p>🔴 High Bullwhip Risk Detected</p>
            <br />
            <ul>
              <li>Increase safety stock</li>
              <li>Use alternate suppliers</li>
              <li>Reduce procurement variability</li>
              <li>Monitor weather continuously</li>
            </ul>
          </>
        )}

        {prediction.prediction === "Medium" && (
          <>
            <p>🟡 Moderate Bullwhip Risk</p>
            <br />
            <ul>
              <li>Reduce unnecessary procurement</li>
              <li>Monitor supplier delays</li>
              <li>Review inventory weekly</li>
              <li>Track demand fluctuations</li>
            </ul>
          </>
        )}

        {prediction.prediction === "Low" && (
          <>
            <p>🟢 Supply Chain Stable</p>
            <br />
            <ul>
              <li>Maintain inventory levels</li>
              <li>Continue monitoring</li>
              <li>Keep supplier communication active</li>
            </ul>
          </>
        )}

      </div>

      <div className="section">

<h2>🧠 Explainable AI</h2>

<br/>

{explanation.top_factors.map((factor,index)=>(

<div
key={index}
style={{
marginBottom:20,
paddingBottom:10,
borderBottom:"1px solid #334155"
}}
>

<h3>{factor.title}</h3>

<p>
<b>Importance:</b> {factor.importance}
</p>

<p>{factor.reason}</p>

</div>

))}

<div className="section">

<h2>📦 Inventory Optimization</h2>

<br/>

<p><b>Status:</b> {inventory.inventory_status}</p>

<p><b>Current Inventory:</b> {inventory.current_inventory}</p>

<p><b>Recommended Inventory:</b> {inventory.recommended_inventory}</p>

<p><b>Holding Cost:</b> ₹{inventory.estimated_holding_cost}</p>

<p><b>Days Remaining:</b> {inventory.days_remaining}</p>

</div>

<div className="section">

<h2>💡 Decision Intelligence</h2>

<br/>

<p>

<b>Priority:</b>{" "}
{recommendation.recommendations[0].priority}

</p>

<br/>

<p>

{recommendation.recommendations[0].reason}

</p>

<br/>

<h3>

Recommended Action

</h3>

<p>

{recommendation.recommendations[0].action}

</p>

</div>

</div>

      <div
        style={{
          textAlign: "center",
          color: "#94a3b8",
          marginTop: 40
        }}
      >

        BullwhipAI • Autonomous Supply Chain Intelligence

      </div>

    </div>

  );

}

export default Dashboard;