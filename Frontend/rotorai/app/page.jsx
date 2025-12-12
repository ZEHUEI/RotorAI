import Hero from '@/Pages/Hero';
import NavigationBar from '@/Pages/NavigationBar';
import ScrollBar from '@/Pages/ScrollBar';

export default function Home() {
  return (
    <>
      <ScrollBar />
      <main className="relative min-h-screen px-1.5 md:px-20 lg:px-40  text-[#D3D1CE]">
        <NavigationBar />
        <Hero />
      </main>
    </>
  );
}
