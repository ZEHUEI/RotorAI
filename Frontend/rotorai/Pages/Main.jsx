'use client';
import { useState } from 'react';
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

//drag and drop

// export default function ImageUpload() {
//   const [file, setFile] = useState(null);
//   const [preview, setPreview] = useState(null);
//   const [result, setResult] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const handleFile = (selectedFile) => {
//     setFile(selectedFile);
//     setPreview(URL.createObjectURL(selectedFile));
//   };

//   const onDrop = (e) => {
//     e.preventDefault();
//     handleFile(e.dataTransfer.files[0]);
//   };

//   const onChange = (e) => {
//     handleFile(e.target.files[0]);
//   };

//   const submit = async () => {
//     if (!file) return;

//     setLoading(true);
//     const formData = new FormData();
//     formData.append("image", file);

//     const res = await fetch("https://your-backend-url/predict", {
//       method: "POST",
//       body: formData,
//     });

//     const data = await res.json();
//     setResult(data);
//     setLoading(false);
//   };

//   return (
//     <div>
//       <div
//         onDrop={onDrop}
//         onDragOver={(e) => e.preventDefault()}
//         onClick={() => document.getElementById("fileInput").click()}
//         style={{
//           border: "2px dashed #aaa",
//           padding: "40px",
//           textAlign: "center",
//           cursor: "pointer",
//         }}
//       >
//         {preview ? (
//           <img src={preview} width={200} />
//         ) : (
//           <p>Drag & drop or click to upload</p>
//         )}
//       </div>

//       <input
//         id="fileInput"
//         type="file"
//         accept="image/*"
//         hidden
//         onChange={onChange}
//       />

//       <button onClick={submit} disabled={loading}>
//         {loading ? "Analyzing..." : "Analyze Image"}
//       </button>

//       {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
//     </div>
//   );
// }

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
            <p className="text-center">
              Drag & Drop Images Here <br /> or <br /> Browse Device
            </p>
          </motion.div>
        </motion.div>
      </div>

      <div className="mt-8"></div>
      <Button />
    </div>
  );
};

export default Main;
