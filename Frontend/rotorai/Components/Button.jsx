'use client';
import { motion } from 'framer-motion';
import React from 'react';

const ButtonVariants = {
  rest: { y: 0 },
  pressed: { y: 10, transition: { duration: 0.2, type: 'tween' } },
};
const ShineVariants = {
  rest: { x: -80, opacity: 0 },
  pressed: {
    x: 200,
    opacity: 1,
    transition: { duration: 0.35, ease: 'easeOut' },
  },
};

//this will activate the backend for flash when press ( on press ) set cpooldown until it is finish

const Button = () => {
  return (
    <>
      <div className="w-[15px] h-full skew-x-30 bg-white"></div>
      <button
        type="button"
        className="relative w-40 h-16 font-AT cursor-pointer border-0 border-amber-300"
      >
        {/*last bottom */}
        <div className="absolute -inset-x-[2px] top-4 -left-[1px] h-full bg-[#3a3a3a] border-2 border-[#525252] rounded-full -z-1" />

        {/* bottom */}
        <div className="absolute top-2.5 left-0 w-full h-full bg-[#0a0a0a] border-2 border-[#525252] rounded-full z-0">
          <div className="absolute w-[2px] h-[9px] bg-[#525252] bottom-0 left-[15%]  " />
          <div className="absolute w-[2px] h-[9px] bg-[#525252] bottom-0 left-[85%]  " />
        </div>

        {/* top animated*/}
        <>
          <motion.div
            variants={ButtonVariants}
            initial="rest"
            whileTap="pressed"
            className="relative z-10 flex items-center justify-center w-full h-full bg-[#0a0a0a] text-lg border-2 border-[#525252] rounded-full overflow-hidden"
          >
            {/* shine */}
            <motion.div
              variants={ShineVariants}
              style={{ skewX: -25 }}
              className="absolute top-0 left-0 w-[15px] h-full bg-white/60 pointer-events-none"
            />
            RUN
          </motion.div>
        </>
      </button>
    </>
  );
};

export default Button;
