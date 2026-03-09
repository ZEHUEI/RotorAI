'use client';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';

const ImageCard = ({ image, title }) => {
  return (
    <>
      <img
        src={image}
        alt={title}
        className="object-cover w-full h-64 rounded-t-lg"
      />
    </>
  );
};

const ButtonHelper = ({}) => {
  return (
    <>
      <div className="border-2 rounded-3xl w-fit px-2 font-Inter text-sm transition-all group-hover:bg-white group-hover:text-black mt-2">
        <div className="p-2">OPEN</div>
      </div>
    </>
  );
};

const BlogPosts = ({ title, image, text, date, postID }) => {
  const router = useRouter();
  return (
    <div
      className="group relative w-full h-[450px] border-0 border-amber-300 rounded-lg bg-[#0a0a0a] backdrop-blur-3xl shadow-xl shadow-white/10 ring-2 ring-white/10 transition-all p-2 hover:cursor-pointer hover:scale-[1.02] transition"
      //this will go to specific shit open
      onClick={() => router.push(`/blog/posts?id=${postID}`)}
    >
      <div className="w-full">
        <ImageCard image={image} title={title} />
        <h2 className="text-xl font-AT mb-2">{title}</h2>
        <span className="text-sm font-ocr">{date}</span>
        <p className="text-sm text-gray-300 font-Inter mt-4 line-clamp-1 lg:line-clamp-2">
          {text}
        </p>
        <div className="absolute bottom-4 right-4">
          <ButtonHelper />
        </div>
      </div>
    </div>
  );
};

export default BlogPosts;
