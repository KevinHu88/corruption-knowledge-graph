import type { Metadata } from 'next'
import './globals.css'
import ToastContainer from '@/components/Toast/ToastContainer'
import { ThemeProvider } from '@/components/Theme/ThemeProvider'

export const metadata: Metadata = {
  title: '腐败知识图谱研判台',
  description: '面向案件图谱与临时文档的证据型知识问答界面',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body>
        <ThemeProvider>
          {children}
          <ToastContainer />
        </ThemeProvider>
      </body>
    </html>
  )
}
