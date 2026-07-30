import {
    Bar
} from "react-chartjs-2";

import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Tooltip,
    Legend
} from "chart.js";

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Tooltip,
    Legend
);

function ProbabilityChart({ probabilities }) {

    const data = {

        labels: ["High", "Medium", "Low"],

        datasets: [

            {

                label: "Prediction Probability",

                data: [

                    probabilities.High,

                    probabilities.Medium,

                    probabilities.Low

                ]

            }

        ]

    };

    const options = {

        responsive: true,

        plugins: {

            legend: {

                display: false

            }

        },

        scales: {

            y: {

                beginAtZero: true,

                max: 1

            }

        }

    };

    return (

        <Bar

            data={data}

            options={options}

        />

    );

}

export default ProbabilityChart;