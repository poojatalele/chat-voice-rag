/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        forest: { DEFAULT: "#065f46", dark: "#064e3b", light: "#d1fae5" },
      },
    },
  },
  plugins: [],
};
