/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          900: '#0B1020',
          800: '#0F1629',
          700: '#1A2342',
        },
        cyan: {
          400: '#22D3EE',
          500: '#06B6D4',
        },
        teal: {
          400: '#2DD4BF',
          500: '#14B8A6',
        },
        amber: {
          500: '#F59E0B',
        },
      },
    },
  },
  plugins: [],
}
