import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = {title:'WindScope — раннее предупреждение об отказах',description:'Исторический SCADA replay с персональной моделью и проверкой ранних предупреждений'};
export default function Layout({children}:{children:React.ReactNode}) {return <html lang="ru"><body>{children}</body></html>;}
