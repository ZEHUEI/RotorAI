import Main from '@/Pages/Main';
import { Analytics } from '@vercel/analytics/react';

export default function Home() {
  return (
    <main className="relative px-1.5 md:px-20 lg:px-40 text-[#D3D1CE]">
      <Analytics />
      <Main />
    </main>
  );
}
