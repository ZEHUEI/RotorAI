'use client';
import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
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
  const [isDragging, setIsDragging] = useState(false);
  const [images, setImages] = useState([]);
  const fileInputRef = useRef(null);
  const MAX_IMAGES = 1;

  const handleFiles = (files) => {
    const imageFiles = Array.from(files).filter((file) =>
      file.type.startsWith('image/')
    );

    setImages((prev) => {
      const remainingSlots = MAX_IMAGES - prev.length;
      if (remainingSlots <= 0) return prev;

      const filesToAdd = imageFiles.slice(0, remainingSlots);

      const previews = filesToAdd.map((file) => ({
        file,
        url: URL.createObjectURL(file),
      }));

      return [...prev, ...previews];
    });
  };

  //drag and drop
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = () => {
    setIsDragging(false);
  };
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  //click and open file
  const handleClick = () => {
    fileInputRef.current?.click();
  };
  const handleFileChange = (e) => {
    handleFiles(e.target.files);
    e.target.value = '';
  };

  return (
    <div className="flex flex-col items-center justify-center font-Inter mt-10">
      <div className="w-full h-[600px]">
        <motion.div
          onClick={images.length < MAX_IMAGES ? handleClick : undefined}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`w-full h-full rounded-xl bg-[#0a0a0a] backdrop-blur-3xl shadow-xl shadow-white/10 flex items-center justify-center ring-2 transition-all   
            ${images.length >= MAX_IMAGES ? 'cursor-not-allowed' : 'cursor-pointer'}
            ${isDragging ? 'ring-white/70 bg-[#111]' : 'ring-white/10'}`}
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/*no image show the text */}
          {images.length === 0 ? (
            <motion.div
              variants={itemVariants}
              className="text-[#525252] text-2xl"
            >
              <p className="text-center">
                Drag & Drop Images Here <br /> or <br /> Browse Device
              </p>
            </motion.div>
          ) : (
            <div
              className="grid grid-cols-1 gap-4 p-6"
              onClick={(e) => e.stopPropagation()}
            >
              {images.map((img, index) => (
                <motion.img
                  key={index}
                  src={img.url}
                  alt={`preview-${index}`}
                  className="w-full h-40 object-cover rounded-lg"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                />
              ))}
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={handleFileChange}
          />
        </motion.div>
      </div>

      <div className="mt-8"></div>
      <Button />
    </div>
  );
};

export default Main;
