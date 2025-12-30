'use client';
import { motion } from 'framer-motion';
import React from 'react';
import Button from '@/Components/Button';

const containerVariants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.2,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0 },
};

const Main = () => {
  return (
    <div className="flex flex-col items-center justify-center font-Inter mt-10">
      <div className="w-full h-[600px]">
        <motion.div
          className="w-full h-full rounded-xl bg-[#0a0a0a] backdrop-blur-3xl shadow-xl shadow-white/10 flex items-center justify-center ring-2 ring-white/10"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div
            variants={itemVariants}
            className="text-[#525252] text-2xl"
          >
            Import or drag ur image here!
          </motion.div>
        </motion.div>
      </div>

      <div className="mt-8"></div>
      <Button />
    </div>
  );
};

export default Main;
