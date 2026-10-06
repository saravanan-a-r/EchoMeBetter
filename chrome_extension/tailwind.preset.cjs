/**
 * EchoMeBetter brand tokens, shared by every Tailwind build (extension pages
 * and the in-page overlay). Classes are prefixed `_echo_$_` so they can never
 * collide with a host page's own utility classes.
 */
module.exports = {
  prefix: '_echo_$_',
  darkMode: 'media',
  theme: {
    extend: {
      colors: {
        echo: {
          50: '#F4F1FF',
          100: '#EAE4FF',
          200: '#D6CBFF',
          300: '#B8A5FF',
          400: '#9677FF',
          500: '#7650FF',
          600: '#6236F5',
          700: '#5128D6',
          800: '#4322AD',
          900: '#371F88',
          950: '#1F1150',
        },
        better: {
          200: '#A7F3E4',
          300: '#5EEAD4',
          400: '#2DD4BF',
          500: '#14B8A6',
          600: '#0D9488',
        },
        ink: {
          50: '#F7F6FB',
          100: '#EEECF5',
          200: '#D9D5E8',
          300: '#B6AFCF',
          400: '#8E86AE',
          500: '#6A6290',
          600: '#4E4672',
          700: '#342B57',
          800: '#231C3D',
          900: '#17122B',
          950: '#0E0B1A',
        },
        danger: { 50: '#FEF2F2', 400: '#F87171', 500: '#E5484D', 600: '#C8363B', 950: '#3B0D10' },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
      },
      backgroundImage: {
        'echo-gradient': 'linear-gradient(135deg, #7650FF 0%, #5B6CFF 45%, #2DD4BF 100%)',
        'echo-glow': 'radial-gradient(60% 60% at 50% 0%, rgba(118, 80, 255, 0.25) 0%, rgba(118, 80, 255, 0) 100%)',
      },
      boxShadow: {
        float: '0 12px 32px -8px rgba(23, 18, 43, 0.45), 0 2px 6px rgba(23, 18, 43, 0.18)',
        card: '0 1px 2px rgba(23, 18, 43, 0.06), 0 8px 24px -12px rgba(23, 18, 43, 0.18)',
      },
      keyframes: {
        'echo-wave': {
          '0%, 100%': { opacity: '0.25', transform: 'scale(0.92)' },
          '50%': { opacity: '1', transform: 'scale(1)' },
        },
        'echo-pop-in': {
          '0%': { opacity: '0', transform: 'translateY(6px) scale(0.98)' },
          '100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
        'echo-shimmer': {
          '0%': { backgroundPosition: '200% 0' },
          '100%': { backgroundPosition: '-200% 0' },
        },
      },
      animation: {
        'echo-wave': 'echo-wave 1.2s ease-in-out infinite',
        'echo-pop-in': 'echo-pop-in 180ms cubic-bezier(0.2, 0.8, 0.2, 1) both',
        'echo-shimmer': 'echo-shimmer 1.6s linear infinite',
      },
    },
  },
  plugins: [],
};
