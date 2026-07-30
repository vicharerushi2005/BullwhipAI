function StatCard({ title, value, icon, color }) {
  return (
    <div
      style={{
        background: "#1e293b",
        borderRadius: 18,
        padding: 20,
        color: "white",
        boxShadow: "0 10px 25px rgba(0,0,0,.25)"
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center"
        }}
      >
        <div>
          <p
            style={{
              color: "#94a3b8",
              marginBottom: 8
            }}
          >
            {title}
          </p>

          <h2
            style={{
              margin: 0,
              color
            }}
          >
            {value}
          </h2>
        </div>

        {icon}
      </div>
    </div>
  );
}

export default StatCard;