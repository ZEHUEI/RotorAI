'use client';
import { useState, useRef } from 'react';
import Button from '@/Components/Button';
import FullBleedDivider from '@/Components/FullBleedDivider';
import axios from 'axios';

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
  const [result, setResult] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [images, setImages] = useState([]);
  const fileInputRef = useRef(null);
  const MAX_IMAGES = 1;
  const isLocked = result !== null;

  //BACKEND
  const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

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

  const handleUpload = async () => {
    if (images.length === 0 || isLocked || isUploading) return;

    setIsUploading(true);

    const formData = new FormData();
    formData.append('image', images[0].file);

    try {
      const res = await axios.post(`${BACKEND_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      //-------------------------------------
      //keep this hidden
      // console.log('backend res', res.data);
      //--------------------------------------
      setResult(res.data);
    } catch (err) {
      console.error('Upload failed', err);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center font-Inter mt-10">
      <div className="w-full h-[600px]">
        <motion.div
          onClick={
            !isLocked && images.length < MAX_IMAGES ? handleClick : undefined
          }
          onDrop={!isLocked ? handleDrop : undefined}
          onDragOver={!isLocked ? handleDragOver : undefined}
          onDragLeave={!isLocked ? handleDragLeave : undefined}
          className={`w-full h-full rounded-xl bg-[#0a0a0a] backdrop-blur-3xl shadow-xl shadow-white/10 flex items-center justify-center ring-2 transition-all   
            ${isLocked ? 'cursor-not-allowed' : 'cursor-pointer'}
            ${isDragging && !isLocked ? 'ring-white/70 bg-[#111]' : 'ring-white/10'}`}
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
                  className="w-full h-60 object-cover rounded-lg"
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
            disabled={isLocked}
          />
        </motion.div>
      </div>

      <div className="mt-8"></div>
      <Button onClick={handleUpload} disabled={isLocked || isUploading} />

      {result && (
        <>
          <div className="w-full h-[0.1px] mt-15 mb-15 bg-white/20" />
          <div className="w-full text-left font-AT text-2xl text-white mb-4">
            Results
          </div>
          <div className="mt-2 grid grid-cols-3 gap-4">
            <div>
              <div className="text-lg text-center font-AT">Original</div>
              <img src={`data:image/png;base64,${result.original_image}`} />
            </div>

            <div>
              <div className="text-lg text-center font-AT">RUST</div>
              <img src={`data:image/png;base64,${result.rust_overlay}`} />
            </div>

            <div>
              <div className="text-lg text-center font-AT">CRACKS</div>
              <img src={`data:image/png;base64,${result.crack_overlay}`} />
            </div>
          </div>
          <FullBleedDivider />
        </>
      )}
    </div>
  );
};

export default Main;
