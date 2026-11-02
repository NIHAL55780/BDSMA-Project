/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#12263a",
        paper: "#f6f1e8",
        accent: "#c45c26",
        teal: "#2a6f6f",
        moss: "#3d5a45",
      },
      fontFamily: {
        serif: ["Source Serif 4", "Georgia", "serif"],
        sans: ["Source Sans 3", "Segoe UI", "sans-serif"],
      },
    },
  },
  plugins: [],
};
