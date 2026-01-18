import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import NavigationBar from '@/Pages/NavigationBar';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata = {
  title: 'RotorAI',
  description: 'Hi This Is RotorAI',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <NavigationBar />
        <div className="pt-20">{children}</div>
      </body>
    </html>
  );
}
