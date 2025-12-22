'use client';
import { motion } from 'framer-motion';
import React from 'react';

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
    <div className="border-2 border-amber-300 w-full h-[750px]">
      <motion.div
        className="flex items-center justify-center h-full"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.div variants={itemVariants} className="text-white text-2xl">
          this will be the part where i insert the image like grok
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Main;
