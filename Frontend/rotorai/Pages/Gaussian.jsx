'use client';
import React, { useEffect, useRef, useState } from 'react';
import { GaussianStore } from '@/lib/gaussianStore';

const Gaussian = () => {
  const containerRef = useRef(null);
  const [isLoading, setIsLoading] = useState(!GaussianStore.isReady());

  useEffect(() => {
    let cancelled = false;

    GaussianStore.init(containerRef.current).then(() => {
      if (cancelled || !containerRef.current) return;

      // Plug the running 3D canvas into the div
      GaussianStore.attach(containerRef.current);

      setIsLoading(false);
    });

    return () => {
      cancelled = true;
      GaussianStore.detach();
    };
  }, []);

  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = ''; // restore on leave
    };
  }, []);

  return (
    <div className="relative flex flex-col overflow-hidden items-center justify-center min-h-screen p-4">
      {/* THE BOX */}

      {/*controls */}
      <div className="absolute top-4 left-1/2 z-30 -translate-x-1/2 text-white font-mono text-xs">
        <div className="grid grid-cols-4 gap-6 bg-black/40 backdrop-blur-sm p-3 rounded-lg border border-white/10">
          {/* Column 1 */}
          <div className="flex flex-col items-center">
            <span className="text-white/40 mb-1">Move</span>
            <span className="tracking-widest">W A S D</span>
          </div>

          {/* Column 2 */}
          <div className="flex flex-col items-center">
            <span className="text-white/40 mb-1">Look</span>
            <span>Right Click</span>
          </div>

          <div className="flex flex-col items-center">
            <span className="text-white/40 mb-1">Rotate</span>
            <span>Left Click</span>
          </div>

          {/* Column 3 */}
          <div className="flex flex-col items-center">
            <span className="text-white/40 mb-1">Zoom</span>
            <span>Scroll</span>
          </div>
        </div>
      </div>

      <div className="relative w-full max-w-[1200px] aspect-video rounded-xl overflow-hidden shadow-2xl shadow-white/10 ring-2 ring-white/10 bg-[#050505]">
        {/* THE CONTAINER: Notice 'overflow-hidden' and 'h-full' */}
        <div
          ref={containerRef}
          className="absolute inset-0 z-10 overflow-hidden h-full w-full"
        />
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black z-20">
            <div className="text-white font-mono animate-pulse">
              LOADING 2064 ENGINE...
            </div>
          </div>
        )}
      </div>
      <p className=" text-white/40 text-xs font-mono uppercase tracking-widest z-30 pointer-events-none mt-2">
        2064 High-Res Splat Engine
      </p>
    </div>
  );
};

export default Gaussian;
